use cubecl_core::{self as cubecl};
use cubecl_core::{ir::StorageType, prelude::*};
use cubecl_runtime::MmaConfig;

use crate::components::error::MatmulSetupError;
use crate::components::tile::TileConfig;
use crate::components::tile::io::TileMut;
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulProblem, TileSize,
    resource::ComputeResources,
    tile::io::{Tile, TileKind},
};
use crate::components::{MatmulElems, MatmulLineSizes, MatmulSelection, MatrixLayout};

/// A family of [TileMatmul] implementations that operate with any [precision](MatmulPrecision).
pub trait TileMatmulFamily: Send + Sync + 'static {
    /// Config for this matmul
    type Config: TileConfig;

    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<L: Numeric, R: Numeric, A: Numeric>: TileMatmul<
            L,
            R,
            A,
            LhsTile = Self::LhsTile,
            RhsTile = Self::RhsTile,
            AccTile = Self::AccTile,
            OutTile = Self::OutTile,
            Config = Self::Config,
        >;

    /// Tile kind for Lhs
    type LhsTile: TileKind;
    /// Tile kind for Rhs
    type RhsTile: TileKind;
    /// Tile kind for Acc
    type AccTile: TileKind;
    /// Tile kind for Out
    type OutTile: TileKind<ReadWrite>;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Whether this matmul family is able to cast on load/store from the stage.
    fn can_cast_stage_element() -> bool;

    /// Returns whether this tile matmul may benefit from swizzling.
    /// Used to determine the selection, since swizzling may require different stage sizes.
    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    /// Returns whether a tile configuration is supported
    fn is_supported<R: Runtime>(_client: &ComputeClient<R>, _config: MmaConfig) -> bool {
        !Self::requires_accelerator()
    }

    /// Returns all sizes supported for these types, if any
    fn supported_sizes<R: Runtime>(
        _client: &ComputeClient<R>,
        _lhs_ty: StorageType,
        _rhs_ty: StorageType,
        _acc_ty: StorageType,
    ) -> Vec<TileSize> {
        Vec::new()
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
    /// Config for this matmul
    type Config: TileConfig;

    /// Contains Lhs data for computation
    type LhsFragment: CubeType;
    /// Contains Rhs data for computation
    type RhsFragment: CubeType;
    /// Contains and accumulates results of the Tile Matmul execution
    type AccFragment: CubeType;

    /// Tile for the lhs data
    type LhsTile: TileKind;
    /// Tile for the rhs data
    type RhsTile: TileKind;
    /// Tile for the accumulator data
    type AccTile: TileKind;
    /// Tile for the output data
    type OutTile: TileKind<ReadWrite>;

    /// Executes the matrix multiplication of Lhs and Rhs, adding the result to the accumulator
    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        out: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    );

    /// Create the container for Lhs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [load_lhs](TileMatmul::load_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment;

    /// Load the container of Lhs from tile data
    fn load_lhs<E: Numeric>(
        tile: &Tile<Self::LhsTile, E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    );

    /// Create the container for Rhs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [load_rhs](TileMatmul::load_rhs) prior to [execute](TileMatmul::execute).
    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment;

    /// Load the container of Rhs from tile data
    fn load_rhs<E: Numeric>(
        tile: &Tile<Self::RhsTile, E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    );

    /// Allocate the container to receive the execution output.
    ///
    /// # Safety
    ///
    /// The output container must be initialized to some value (typically 0),
    /// because the execution adds to the already present value.
    /// Make sure to call [load_acc](TileMatmul::load_acc) prior to [execute](TileMatmul::execute).
    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment;

    /// Load the container of Acc from tile data
    fn load_acc<E: Numeric>(
        tile: &Tile<Self::AccTile, E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    );

    /// Write the content of the output container to the given slice
    fn write_results<E: Numeric>(
        tile: &mut TileMut<Self::OutTile, E>,
        out: &Self::AccFragment,
        #[comptime] config: Self::Config,
    );
}
