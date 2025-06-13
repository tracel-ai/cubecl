use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, ir::Elem};

use crate::{
    components::{
        AvailableLineSizes, Ident, InvalidConfigError, MatmulChecker, MatmulPrecision,
        MatmulProblem, MatrixLayout, TileSize, config::MatmulConfig, resource::ComputeResources,
        stage::StageVectorization, tile::tile_data::Tile,
    },
    kernels::{MatmulSetupError, matmul::MatmulSelection},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TileSetupInput {
    pub vectorization: StageVectorization,
    pub tile_size: TileSize,
}

pub trait TileMatmulFamily: Send + Sync + 'static + MatmulChecker<Config: TileConfig> {
    type Matmul<MP: MatmulPrecision>: TileMatmul<MP, Config = Self::Config>;

    fn requires_tensor_cores() -> bool;
    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection;
}

/// Provides matrix multiplication operations at the tile level.
///
/// At the tile level,
///  - Dimensions M, N and K are fixed to an integer, and the
///    matrix multiplication works only for size (M, K) · (K, N) = (M, N).
///
/// Assumptions:
///  - Inputs must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
///  - Enough units are present to perform the whole computation
#[cube]
pub trait TileMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: TileConfig;
    /// Contains LHS data that can be split across the units
    type Lhs: CubeType;
    /// Contains RHS data that can be split across the units
    type Rhs: CubeType;
    /// Contains output data that can be split across the units
    type Accumulator: CubeType;

    /// Executes the matrix multiplication of LHS and RHS, adding the result to the output
    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Create the container for LHS data
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_lhs](TileMatmul::fill_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs;

    /// Fill the container of LHS with data
    fn fill_lhs(slice: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config);

    /// Create the container for RHS data
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_rhs](TileMatmul::fill_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;

    /// Fill the container of RHS with data
    fn fill_rhs(slice: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config);

    /// Allocate the container to receive the execution output.
    ///
    /// # Safety
    ///
    /// The output container must be initialized to some value (typically 0),
    /// because the execution adds to the already present value.
    /// Make sure to call either [fill_accumulator](TileMatmul::fill_accumulator)
    /// or [zero_accumulator](TileMatmul::zero_accumulator).
    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with data
    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Fill the accumulator with zeros.
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);

    /// Write the content of the output container to the given slice
    fn write_results(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<MP::EO>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile matmul (TMM) level
pub trait TileConfig: MatmulConfig {
    /// Returns the size of the plane dimension
    fn plane_dim(&self) -> u32;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout;

    /// Returns the line size for the given ident
    fn stage_line_size<I: Into<Ident>>(&self, ident: I) -> u32;

    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32;

    /// Returns the shape of the tiles in the three axes m, k and n.
    fn tile_size(&self) -> &TileSize;
}
