use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatrixLayout,
    TilingDimensions,
    config::MatmulConfig,
    stage::{self, Writer},
    tile,
};
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use super::Quantization;

/// A family of [matmuls](GlobalMatmul) working with any [precision](MatmulPrecision).
pub trait GlobalMatmulFamily:
    MatmulConfigFactory<Config: GlobalConfig> + Send + Sync + 'static
{
    type Matmul<MP: MatmulPrecision>: GlobalMatmul<MP, Config = Self::Config>;
}

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
pub trait GlobalMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: GlobalConfig;
    type LhsLoader: CubeType;
    type RhsLoader: CubeType;
    type AccumulatorLoader: CubeType;
    type Out: Writer<MP::EO>;
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
        lhs: VirtualTensor<MP::EI>,
        m_offset: u32,
        k_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader;

    /// Initialize the loader for Rhs, starting at row k and column n
    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        k_offset: u32,
        n_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader;

    /// Initialize the unloader at row m and column n
    fn init_unloader(
        out: VirtualTensor<MP::EO, ReadWrite>,
        m_offset: u32,
        n_offset: u32,
        nth_batch: u32,
        batch_offset: u32,
    ) -> Self::Out;

    /// Initialize the accumulator without data
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with zeros
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);
}

#[cube]
/// Input to the global matmul accumulator, responsible of filling the stage and providing a reader
/// for it.
pub trait AccumulatorLoader<MP: MatmulPrecision>: CubeType + 'static + Send + Sync {
    fn fill_stage<G: GlobalConfig>(this: &mut Self, #[comptime] config: G);

    /// Load accumulator for `nth_tile`. Should call either `zero_accumulator` or `fill_accumulator`
    /// for the underlying tile.
    fn load<Tile: tile::TileMatmul<MP>>(
        this: &mut Self,
        acc: &mut Tile::Accumulator,
        nth_tile: u32,
        #[comptime] config: Tile::Config,
    );
}

pub trait LoadingValidation {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError>;
}

/// Configuration for the [global matmul](GlobalMatmul) level.
pub trait GlobalConfig: MatmulConfig {
    /// Underlying Stage matmul config
    type SmmConfig: stage::StageConfig;

    /// Convert itself to the underlying stage matmul config
    fn to_smm_config(&self) -> Self::SmmConfig;

    /// Returns the line size for the global memory corresponding to the given ident
    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32;

    /// Returns the [StageTiling] for the given ident
    fn tiling_dimensions<I: Into<Ident>>(&self, ident: I) -> TilingDimensions;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout;

    /// Returns the number of planes in the cube
    fn num_planes(&self) -> u32;

    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Whether to check if accessing a row would exceed bounds.
    fn check_row_bounds<I: Into<Ident>>(&self, ident: I) -> bool;

    /// Whether to check if accessing a col would exceed bounds.
    fn check_col_bounds<I: Into<Ident>>(&self, ident: I) -> bool;

    /// Whether to check if accessing a col for lhs or row for rhs would exceed bounds.
    fn check_k_bounds(&self) -> bool;

    fn precompute_job(&self) -> bool;

    fn num_stages(&self, ident: InputIdent) -> u32;
}
