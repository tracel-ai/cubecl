use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::stage::{self, StageReader, StageWriter, TilingOrderConfig};
use crate::matmul::components::MatmulKernel;
use crate::matmul::components::StageDim;
use crate::matmul::components::{Ident, MatrixLayout};

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
    type Lhs: Loader<EG, ES>;
    type Rhs: Loader<EG, ES>;
    type Out: Unloader<EG>;

    /// Performs the matrix multiplication over data loaded by the
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output unloader.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_loader: Self::Lhs,
        rhs_loader: Self::Rhs,
        unloader: Self::Out,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );
}

#[cube]
/// Input to the global matmul, responsible of filling the stage and providing a reader for it.
/// Advances along the k-dimension to fill the stage with further data.
pub trait Loader<EG: Numeric, ES: Numeric>: CubeType + 'static + Send + Sync {
    /// The stage reader which matches the input of the underlying stage matmul.
    type StageReader: StageReader<ES>;

    /// Fills the stage at the current k offset and returns a reader for it.
    fn fill_stage<G: Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader;

    /// Move the k offset by k_offset
    fn advance_view(this: &mut Self, k_offset: u32);

    fn new<G: Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self;
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

    fn new(tensor: &mut Tensor<Line<EG>>, x_offset: u32, y_offset: u32, batch_offset: u32) -> Self;
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
    fn stage_dim(&self, ident: Ident) -> StageDim;

    /// Returns the [MatrixLayout] for the given ident
    fn layout(&self, ident: Ident) -> MatrixLayout;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the order in which tiles should be loaded to the stage
    fn tiling_order(&self) -> TilingOrderConfig;

    /// Whether it is necessary to add bound checks in the m dimension
    fn check_m_bounds(&self) -> bool;

    /// Whether it is necessary to add bound checks in the n dimension
    fn check_n_bounds(&self) -> bool;

    /// Whether we transpose data when loading to the stage
    fn transpose_load(&self, ident: Ident) -> bool;
}
