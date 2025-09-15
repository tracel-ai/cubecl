use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulProblem, MatrixLayout, TileSize,
    resource::ComputeResources, tile::loader::TileKind,
};
use crate::components::{MatmulLineSizes, MatmulSelection};
use crate::components::{StageIdent, tile::loader::Loader};
use crate::components::{error::MatmulSetupError, tile::loader::LoaderTile};
use std::{fmt::Debug, hash::Hash};

/// A family of [TileMatmul] implementations that operate with any [precision](MatmulPrecision).
pub trait TileMatmulFamily: Send + Sync + 'static {
    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<L: Numeric, R: Numeric, A: Numeric>: TileMatmul<
            L,
            R,
            A,
            Config = Self::Config,
            LhsLoader: Loader<TileKind = Self::LhsTile>,
            RhsLoader: Loader<TileKind = Self::RhsTile>,
            AccLoader: Loader<TileKind = Self::AccTile>,
        >;

    /// Tile kind for Lhs
    type LhsTile: TileKind;
    /// Tile kind for Rhs
    type RhsTile: TileKind;
    /// Tile kind for Acc
    type AccTile: TileKind;

    /// The configuration type associated with this matmul family.
    type Config: TileConfig;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
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
pub trait TileMatmul<L: Numeric, R: Numeric, A: Numeric>: 'static + Send + Sync {
    /// The configuration type associated with this Matmul.
    type Config: TileConfig;
    /// Contains Lhs data for computation
    type Lhs: CubeType;
    /// Contains Rhs data for computation
    type Rhs: CubeType;
    /// Contains and accumulates results of the Tile Matmul execution
    type Accumulator: CubeType;

    /// Loader for the lhs data
    type LhsLoader: Loader;
    /// Loader for the rhs data
    type RhsLoader: Loader;
    /// Loader for the accumulator data
    type AccLoader: Loader;

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
    fn fill_lhs<E: Numeric>(
        tile: LoaderTile<Self::LhsLoader, E>,
        lhs: &mut Self::Lhs,
        #[comptime] config: Self::Config,
    );

    /// Create the container for Rhs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [fill_rhs](TileMatmul::fill_rhs) prior to [execute](TileMatmul::execute).
    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;

    /// Fill the container of Rhs with tile data
    fn fill_rhs<E: Numeric>(
        tile: LoaderTile<Self::RhsLoader, E>,
        rhs: &mut Self::Rhs,
        #[comptime] config: Self::Config,
    );

    /// Allocate the container to receive the execution output.
    ///
    /// # Safety
    ///
    /// The output container must be initialized to some value (typically 0),
    /// because the execution adds to the already present value.
    /// Make sure to call either [fill_accumulator](TileMatmul::fill_accumulator)
    /// or [zero_accumulator](TileMatmul::zero_accumulator).
    fn allocate_acc(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with data
    fn fill_acc<E: Numeric>(
        tile: LoaderTile<Self::AccLoader, E>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Write the content of the output container to the given slice
    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
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
