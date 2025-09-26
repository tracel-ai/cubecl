use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::layout::Coords2d};

use crate::components::{AccS, global::MaxGlobalReaderPlanes};
use crate::components::{
    AvailableLineSizes, LhsS, MatmulLineSizes, MatmulSelection, RhsS, StageIdent,
};
use crate::components::{
    MatmulPrecision, MatmulProblem, MatrixLayout, TilingScheme,
    global::{self, PlaneRoleConfig, RoleRuleConfig},
    tile::TileConfig,
};
use crate::components::{
    error::MatmulSetupError, global::WriteEventListener, stage::StageMemoryConfig,
};
use crate::components::{
    stage::{NumStages, PartitionScheduler, PartitionSchedulerScheme},
    tile::io::TileKind,
};
use std::{fmt::Debug, hash::Hash};

use super::{StageEventListener, TilingLayout};

/// A family of [StageMatmul] implementations that operate with any [precision](MatmulPrecision).
pub trait StageMatmulFamily: Send + Sync + 'static {
    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout, TA: TilingLayout, TO: TilingLayout>: StageMatmul<
            MP,
            Config = Self::Config,
            LhsStage = <Self::LhsStage as StageFamily>::Stage<LhsS<MP>, TL>,
            RhsStage = <Self::RhsStage as StageFamily>::Stage<RhsS<MP>, TR>,
            AccStage = <Self::AccStage as StageFamily>::Stage<AccS<MP>, TA>,
            OutStage = <Self::OutStage as StageFamily<ReadWrite>>::Stage<AccS<MP>, TO>,
        >;

    /// Stage family for Lhs
    type LhsStage: StageFamily;
    /// Stage family for Rhs
    type RhsStage: StageFamily;
    /// Stage family for Acc
    type AccStage: StageFamily;
    /// Stage family for Out
    type OutStage: StageFamily<ReadWrite>;

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
        max_global_readers: Option<MaxGlobalReaderPlanes>,
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
    type Accumulators: CubeType;

    /// Stage for Lhs
    type LhsStage: CubeType;
    /// Stage for Rhs
    type RhsStage: CubeType;
    /// Stage for Accumulator
    type AccStage: CubeType;
    /// Stage for Out
    type OutStage: CubeType;

    /// Lhs input of the underlying Tile Matmul
    type LhsTile: CubeType;
    /// Rhs input of the underlying Tile Matmul
    type RhsTile: CubeType;

    /// Executes the matrix multiplication of Lhs and Rhs, adding the result to the accumulator
    ///
    /// Equivalent to execute_with_listener with SEL:=NoEvent
    fn execute(
        lhs: &Self::LhsStage,
        rhs: &Self::RhsStage,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        partition_scheduler: &PartitionScheduler,
    );

    /// Executes the matrix multiplication of Lhs and Rhs, with the addition of injected
    /// [event listener](StageEventListener).
    fn execute_with_listener<SEL: StageEventListener<Self::Config>>(
        lhs: &Self::LhsStage,
        rhs: &Self::RhsStage,
        instruction_lhs: &mut Self::LhsTile,
        instruction_rhs: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        listener: SEL,
        partition_scheduler: &PartitionScheduler,
    );

    /// Inits inputs of the underlying Tile Matmul
    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile);

    /// Create an instance of the accumulators, without data
    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators;

    /// Load all accumulators in the stage from data
    fn load_accumulators(
        reader: &Self::AccStage,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
    );

    /// Reads the result of the accumulator and hands it to the stage writer
    fn write_results<W: WriteEventListener, G: global::GlobalConfig>(
        acc: &Self::Accumulators,
        stage: &mut Self::OutStage,
        listener: &mut W,
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

    /// Converts itself to the underlying Tile Matmul config
    fn tile_config(self) -> Self::TileConfig;

    /// Converts itself to the underlying Stage Memory config
    fn stage_memory_config(self, ident: StageIdent) -> StageMemoryConfig {
        let tiling = self.tiling_scheme();
        StageMemoryConfig {
            num_main_flow_planes: self.num_main_flow_planes(),
            elements_in_tile_row: tiling.elements_in_tile_row(ident),
            elements_in_tile_col: tiling.elements_in_tile_col(ident),
            tiles_in_stage_row: tiling.tiles_in_stage_row(ident),
            tiles_in_stage_col: tiling.tiles_in_stage_col(ident),
            stage_line_size: self.stage_line_size(ident),
            matrix_layout: self.matrix_layout(ident),
            num_stages: self.num_stages(ident),
        }
    }

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

    /// Number of stages in the stage
    fn num_stages(&self, ident: StageIdent) -> u32;
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PartitionBuffering {
    Single,
    #[default]
    Double,
}

/// Stage that can be divided into tiles, with the same kind used by the
/// tile matmul readers.
#[cube]
pub trait Stage<ES: Numeric, IO: SliceVisibility = ReadOnly>:
    CubeType + Send + Sync + 'static
{
    /// The kind (or family) of the tiles contained in this stage
    type TileKind: TileKind<IO>;

    /// Slices a tile with offset (`row`, `col`) from the stage and returns it
    fn tile(this: &Self, tile: Coords2d) -> <Self::TileKind as TileKind<IO>>::Tile<ES>;
}

/// Stage family for any precision
pub trait StageFamily<IO: SliceVisibility = ReadOnly>: Send + Sync + 'static {
    /// The tile kind (family) contained in the stage
    type TileKind: TileKind<IO>;
    /// The concrete stage type of this family, instantiated with the type and layout
    type Stage<ES: Numeric, T: TilingLayout>: Stage<ES, IO, TileKind = Self::TileKind>;
}

#[cube]
impl<ES: Numeric, IO: SliceVisibility, Inner: Stage<ES, IO>> Stage<ES, IO> for CubeOption<Inner> {
    type TileKind = CubeOption<Inner::TileKind>;

    fn tile(this: &Self, tile: Coords2d) -> <Self::TileKind as TileKind<IO>>::Tile<ES> {
        match this {
            CubeOption::Some(stage) => CubeOption::new_Some(Inner::tile(stage, tile)),
            CubeOption::None => CubeOption::new_None(),
        }
    }
}

impl<IO: SliceVisibility, Inner: StageFamily<IO>> StageFamily<IO> for Option<Inner> {
    type TileKind = CubeOption<Inner::TileKind>;
    type Stage<ES: Numeric, T: TilingLayout> = CubeOption<Inner::Stage<ES, T>>;
}
