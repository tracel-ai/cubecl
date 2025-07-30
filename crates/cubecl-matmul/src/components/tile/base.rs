use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::StageIdent;
use crate::components::error::MatmulSetupError;
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    MatmulSelection, MatrixLayout, TileSize, resource::ComputeResources, tile::tile_data::Tile,
};
use std::{fmt::Debug, hash::Hash};

/// A family of [TileMatmul] implementations that operate with any [precision](MatmulPrecision).
pub trait TileMatmulFamily: Send + Sync + 'static {
    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision>: TileMatmul<MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: TileConfig;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tile_setup_info: TileSetupInfo,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
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
    /// The configuration type associated with this Matmul.
    type Config: TileConfig;
    /// Contains Lhs data for computation
    type Lhs: CubeType;
    /// Contains Rhs data for computation
    type Rhs: CubeType;
    /// Contains and accumulates results of the Tile Matmul execution
    type Accumulator: CubeType;

    /// Executes the matrix multiplication of Lhs and Rhs, adding the result to the accumulator
    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Create the container for Lhs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_lhs](TileMatmul::fill_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs;

    /// Fill the container of Lhs with tile data
    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config);

    /// Create the container for Rhs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_rhs](TileMatmul::fill_rhs) prior to [execute](TileMatmul::execute).
    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;

    /// Fill the container of Rhs with tile data
    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config);

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

/// Configuration for the Tile Matmul level
pub trait TileConfig: Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static {
    /// Returns the number of units in a plane
    fn plane_dim(&self) -> u32;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout;

    /// Returns the line size for the given ident
    fn stage_line_size(&self, ident: StageIdent) -> u32;

    /// Returns the line size for the given ident
    fn global_line_size(&self, ident: StageIdent) -> u32;

    /// Returns the (m,n,k) shape of the tiles
    fn tile_size(&self) -> &TileSize;
}

pub struct TileSetupInfo {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u32,
    pub rhs_line_size: u32,
    pub out_line_size: u32,
}

impl TileSetupInfo {
    pub fn from_matmul(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Self {
        Self {
            tile_size: selection.tiling_scheme.tile_size,
            plane_dim: selection.plane_dim,
            lhs_layout: problem.lhs_layout,
            rhs_layout: problem.rhs_layout,
            lhs_line_size: line_sizes.lhs as u32,
            rhs_line_size: line_sizes.rhs as u32,
            out_line_size: line_sizes.out as u32,
        }
    }
}
