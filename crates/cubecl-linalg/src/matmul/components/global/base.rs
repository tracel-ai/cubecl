use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self, StageWriter, TilingOrderConfig};
use crate::matmul::components::MatmulKernel;
use crate::matmul::components::StageDim;
use crate::matmul::components::{config::MatmulConfig, tile};
use crate::matmul::components::{Ident, MatrixLayout};

use super::tensor_view::TensorReader;

#[cube]
/// Provides matrix multiplication operations at the global level.
///
/// At the global level,
///  - Inputs are views over global memory, meaning access is given to
///    only parts of the global memory inputs at once.
///  - All planes within a Cube can collaborate to solve the problem
///  - Dimensions M and N are fixed to an integer, but K is arbitrary large.
///    The matrix multiplication works only for size (M, _) · (_, N) = (M, N).
///    M and N should match the underlying Stage matmul's M and N.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
///
/// # Safety
///
/// It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
/// It is therefore important that Loaders and Unloaders perform checks to avoid out-of-bounds
/// before loading data.
pub trait Matmul<EG: Numeric, ES: Numeric>:
    'static + Send + Sync + MatmulKernel<EG, EG, Config: Config>
{
    type LhsLoader: Loader<EG, ES>;
    type RhsLoader: Loader<EG, ES>;
    type AccumulatorLoader: CubeType;
    type Out: Unloader<EG>;
    type Accumulator: CubeType;

    /// Performs the matrix multiplication over data loaded by the
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output unloader.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_loader: Self::LhsLoader,
        rhs_loader: Self::RhsLoader,
        unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    /// Initialize the loader for Lhs, starting at row m and column k
    fn init_lhs_loader(
        lhs: &Tensor<Line<EG>>,
        m_offset: u32,
        k_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader;

    /// Initialize the loader for Rhs, starting at row k and column n
    fn init_rhs_loader(
        rhs: &Tensor<Line<EG>>,
        k_offset: u32,
        n_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader;

    /// Initialize the unloader at row m and column n
    fn init_unloader(
        out: &mut Tensor<Line<EG>>,
        m_offset: u32,
        n_offset: u32,
        batch_offset: u32,
    ) -> Self::Out;

    /// Initialize the accumulator without data
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with zeros
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);
}

#[derive(CubeType)]
pub struct LoadBuffer<EG: Numeric> {
    pub array: Array<Line<EG>>,
    pub length: u32,
}

#[cube]
impl<EG: Numeric> LoadBuffer<EG> {
    pub fn slice(&self, start: u32, end: u32) -> Slice<Line<EG>> {
        self.array.slice(start, end)
    }

    pub fn slice_mut(&mut self, start: u32, end: u32) -> SliceMut<Line<EG>> {
        self.array.slice_mut(start, end)
    }
}

#[cube]
/// Input to the global matmul, responsible of filling the stage and providing a reader for it.
/// Advances along the k-dimension to fill the stage with further data.
pub trait Loader<EG: Numeric, ES: Numeric>: CubeType + 'static + Send + Sync {
    /// The stage reader which matches the input of the underlying stage matmul.
    type StageReader: CubeType;

    fn init_buffer<G: Config>(#[comptime] config: G) -> LoadBuffer<EG>;

    fn fetch_global<G: Config>(this: &Self, buffer: &mut SliceMut<Line<EG>>, #[comptime] config: G);

    fn fill_stage<G: Config>(
        this: &mut Self,
        buffer: &Slice<Line<EG>>,
        #[comptime] config: G,
    ) -> Self::StageReader;

    fn to_next_stage<G: Config>(this: &mut Self, #[comptime] config: G);
}

#[cube]
pub trait LoadingStrategy<EG: Numeric, ES: Numeric>: 'static + Send + Sync + Clone {
    fn init_buffer<G: Config>(#[comptime] ident: Ident, #[comptime] config: G) -> LoadBuffer<EG>;

    fn fetch<G: Config>(
        read_view: &TensorReader<EG>,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );

    fn store<G: Config>(
        buffer: &Slice<Line<EG>>,
        stage_slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

#[cube]
/// Input to the global matmul accumulator, responsible of filling the stage and providing a reader
/// for it.
pub trait AccumulatorLoader<O: Numeric, Acc: Numeric>: CubeType + 'static + Send + Sync {
    fn fill_stage<G>(this: &mut Self, #[comptime] config: G);

    /// Load accumulator for `tile_n`. Should call either `zero_accumulator` or `fill_accumulator`
    /// for the underlying tile.
    fn load<I: Numeric, Tile: tile::Matmul<I, Acc>>(
        this: &mut Self,
        acc: &mut Tile::Accumulator,
        tile_n: u32,
        #[comptime] config: Tile::Config,
    );
}

#[cube]
/// Output to the global matmul
///
/// # Note
///
/// It is only a wrapper over the stage writer because there is no K for the output.
/// Could be deleted in favor of having only the StageWriter
pub trait Unloader<EG: Numeric>: CubeType + 'static + Send + Sync {
    type StageWriter: StageWriter<EG>;

    fn as_stage_writer<G: Config>(unloader: Self) -> Self::StageWriter;
}

/// Configuration for the Global matmul (GMM) level
pub trait Config: MatmulConfig {
    /// Underlying Stage matmul config
    type SmmConfig: stage::Config;

    /// Convert itself to the underlying stage matmul config
    fn to_smm_config(&self) -> Self::SmmConfig;

    /// Returns the line size for the global memory corresponding to the given ident
    fn global_line_size(&self, ident: Ident) -> u32;

    /// Returns the line size for the stage of the given ident
    fn stage_line_size(&self, ident: Ident) -> u32;

    /// Returns the [StageDim] for the given ident
    fn stage_dim(&self, ident: Ident) -> Box<dyn StageDim>;

    /// Returns the [MatrixLayout] for the given ident
    fn layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the order in which tiles should be loaded to the stage
    fn tiling_order(&self, ident: Ident) -> TilingOrderConfig;

    /// Whether it is necessary to add bound checks in the m dimension
    fn check_m_bounds(&self) -> bool;

    /// Whether it is necessary to add bound checks in the k dimension
    fn check_k_bounds(&self) -> bool;

    /// Whether it is necessary to add bound checks in the n dimension
    fn check_n_bounds(&self) -> bool;

    /// Whether we transpose data when loading to the stage
    fn transpose_load(&self, ident: Ident) -> bool;

    /// How many buffers (single/double buffering)
    fn num_buffers(&self) -> u32;
}
