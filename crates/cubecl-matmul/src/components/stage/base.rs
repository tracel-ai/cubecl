use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coordinates};

use crate::components::error::MatmulSetupError;
use crate::components::global::MaxLoaderPlanes;
use crate::components::stage::{
    NumStages, PartitionScheduler, PartitionSchedulerScheme, StageMemoryConfig,
};
use crate::components::tile::Tile;
use crate::components::{
    AvailableLineSizes, LhsS, MatmulLineSizes, MatmulSelection, RhsS, StageIdent,
};
use crate::components::{
    MatmulPrecision, MatmulProblem, MatrixLayout, TilingScheme,
    global::{self, AccumulatorLoader, GlobalWriter, PlaneRoleConfig, RoleRuleConfig},
    tile::TileConfig,
};
use std::{fmt::Debug, hash::Hash};

use super::{StageEventListener, TilingLayout};

/// A family of [StageMatmul] implementations that operate with any [precision](MatmulPrecision).
pub trait StageMatmulFamily: Send + Sync + 'static {
    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout>: StageMatmul<
            MP,
            Config = Self::Config,
            LhsReader = <Self::LhsReader as ReaderFamily>::Reader<LhsS<MP>, TL>,
            RhsReader = <Self::RhsReader as ReaderFamily>::Reader<RhsS<MP>, TR>,
            WriteCoords = Self::WriteCoords,
        >;

    /// Reader family for Lhs
    type LhsReader: ReaderFamily;
    /// Reader family for Rhs
    type RhsReader: ReaderFamily;
    /// Writer coordinate type
    type WriteCoords: Coordinates;

    /// The configuration type associated with this matmul family.
    type Config: StageConfig;

    /// Constructs the configuration based on the matmul problem, selection, line sizes,
    /// number of stages, maximum of tasks per plane, and whether the algorithm is an ordered variant
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_loaders: Option<MaxLoaderPlanes>,
        ordered: bool,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
}

#[cube]
/// Provides matrix multiplication operations at the stage level.
///
/// At the stage level,
///  - Inputs are assumed to be already staged into a shared memory.
///  - All main flow planes within a Cube are used to solve the problem
///  - Dimensions M, N and K are fixed to an integer, and the
///    matrix multiplication works only for size (M, K) · (K, N) = (M, N).
///    These integers are multiples of the underlying Tile matmul,
///    corresponding to the number of tiles in each dimension.
///
/// Assumptions:
///  - Data given as inputs by stage readers must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
///  - Enough planes/units are launched to perform the whole computation
pub trait StageMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    /// The configuration type associated with this Matmul.
    type Config: StageConfig;

    /// Contains the matrix multiplication output, that can be shared across the different planes of the cube.
    /// The same Accumulator will be added to across multiple executions of the Stage Matmul.
    type Accumulator: CubeType;

    /// How to read shared memory for Lhs
    type LhsReader: CubeType;
    /// How to read shared memory for Rhs
    type RhsReader: CubeType;

    /// Lhs input of the underlying Tile Matmul
    type LhsTile: CubeType;
    /// Rhs input of the underlying Tile Matmul
    type RhsTile: CubeType;

    /// How to write to global memory after computation
    type Writer: GlobalWriter<MP::EO, Coordinates = Self::WriteCoords>;
    /// Coordinates used by the writer
    type WriteCoords: Coordinates;

    /// Executes the matrix multiplication of Lhs and Rhs, adding the result to the accumulator
    ///
    /// Equivalent to execute_with_listener with SEL:=NoEvent
    fn execute(
        lhs: &Self::LhsReader,
        rhs: &Self::RhsReader,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
        partition_scheduler: &PartitionScheduler,
    );

    /// Executes the matrix multiplication of Lhs and Rhs, with the addition of injected
    /// [event listener](StageEventListener).
    fn execute_with_listener<SEL: StageEventListener<Self::Config>>(
        lhs: &Self::LhsReader,
        rhs: &Self::RhsReader,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
        listener: SEL,
        partition_scheduler: &PartitionScheduler,
    );

    /// Inits inputs of the underlying Tile Matmul
    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile);

    /// Create an instance of the accumulator, without data
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    /// Fill the accumulator with zeros
    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config);

    /// Fill the accumulator with data
    fn fill_accumulator<L: AccumulatorLoader<MP>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    /// Inits the writer at the given offsets
    fn init_writer(
        tensor: View<Line<MP::EO>, Self::WriteCoords, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer;

    /// Reads the result of the accumulator and hands it to the stage writer
    fn write_results<G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut Self::Writer,
        partition_scheduler: &PartitionScheduler,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    );

    fn init_scheduler(#[comptime] config: Self::Config) -> PartitionScheduler;
}

/// Configuration for the Stage matmul (SMM) level
pub trait StageConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    /// Underlying Tile matmul config
    type TileConfig: TileConfig;
    type StageMemoryConfig: StageMemoryConfig;

    /// Converts itself to the underlying Tile Matmul config
    fn tile_config(self) -> Self::TileConfig;

    /// Converts itself to the underlying Stage Memory config
    fn stage_memory_config(self) -> Self::StageMemoryConfig;

    /// Returns the line size for the given ident
    fn stage_line_size(&self, ident: StageIdent) -> u32;

    /// Returns the line size for the given ident
    fn global_line_size(&self, ident: StageIdent) -> u32;

    /// Returns the [MatrixLayout] for the given ident
    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout;

    /// Returns how many units are in a plane
    fn plane_dim(&self) -> u32;

    /// Returns whether we must perform partition buffering
    fn partition_buffering(&self) -> PartitionBuffering;

    /// Returns the [TilingScheme]
    fn tiling_scheme(&self) -> TilingScheme;

    /// Indicates the specialization roles for the planes
    fn plane_role_config(&self) -> PlaneRoleConfig;

    /// How to identify the role of the plane depending on its index
    fn role_rule_config(&self) -> RoleRuleConfig;

    /// Number of planes participating in the main computation flow
    fn num_main_flow_planes(&self) -> u32;

    /// Whether the Matmul is quantized
    fn quantized(&self) -> bool;

    /// Whether we must sync planes after execution because the execution
    /// is not sync by itself (depends on the runtime/compiler)
    fn must_sync_plane_after_execution(&self) -> bool;

    fn partition_schedule_scheme(&self) -> PartitionSchedulerScheme;
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PartitionBuffering {
    Single,
    #[default]
    Double,
}

#[cube]
/// Read the tile at (row, col) from stage memory
pub trait StageToTileReader<ES: Numeric>: CubeType + Send + Sync + 'static {
    fn read_tile<S: StageMemoryConfig>(
        this: &Self,
        row: u32,
        col: u32,
        #[comptime] config: S,
    ) -> Tile<ES>;
}

/// Reader family for any precision
pub trait ReaderFamily: Send + Sync + 'static {
    type Reader<ES: Numeric, T: TilingLayout>: StageToTileReader<ES>;
}
