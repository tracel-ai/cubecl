use crate::components::stage::{StridedStageMemory, SwizzleMode};
use crate::components::{InvalidConfigError, StageIdent};
use crate::components::{
    LhsS,
    global::{GlobalConfig, GlobalReaderConfig},
};
use crate::components::{MatmulElems, global::read::AsyncPartialLoadingStrategy};
use crate::components::{
    MatmulPrecision,
    global::read::{validate_async_barrier, validate_tma},
};
use crate::components::{MatmulProblem, stage::TmaTilingLayout};
use crate::components::{
    MatrixLayout,
    global::read::{PartialLoadingStrategy, async_tma::AsyncTma},
};
use crate::components::{
    RhsS,
    global::{RoleRule, multi_stage::LoadMaxRoundPlaneCount},
};
use crate::components::{
    global::SharedGlobalMatmulConfig,
    stage::{StageConfig, StridedStageFamily},
};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using TMA load instructions.
/// Uses special tiling to minimize the number of loads required. Issues one load for each
/// tile in the major dimension (i.e. `k` for col-major RHS).
pub struct AsyncPartialTmaLoading {}

impl LoadingValidation for AsyncPartialTmaLoading {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        TmaTilingLayout::check(config.smem_config)?;
        validate_tma(client, problem, config, dtypes)?;

        validate_async_barrier(client)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialTmaLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        _tiles_per_stage: u32,
        _line_size: u8,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        4
    }
}

#[cube]
impl PartialLoadingStrategy for AsyncPartialTmaLoading {
    type TilingLayout = TmaTilingLayout;
    type SyncStrategy = AsyncTma;
    type Stage = StridedStageFamily;

    type Job<EG: Numeric, ES: Numeric> = AsyncPartialTmaJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
        #[comptime] _line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let role_rule_config = config.plane_role_config.rule;
        let config = config.smem_config;
        let tile_count_col = match config.matrix_layout {
            MatrixLayout::RowMajor => config.tiles_per_stage_along_col(),
            MatrixLayout::ColMajor => config.tiles_per_stage_along_row(),
        };
        // Swizzle renders the column format irrelevant, so we load the whole stage at once
        // The tiling is set on launch for TMA, so no further change is needed here.
        let num_tasks = comptime![match config.swizzle {
            SwizzleMode::None => tile_count_col,
            _ => 1u32,
        }];

        let is_elected = RoleRule::new(role_rule_config).elect_load_leader();

        AsyncPartialTmaJob {
            is_elected,
            num_tasks,
            stage_index,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialTmaJob {
    is_elected: bool,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    stage_index: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, TmaTilingLayout, AsyncTma>
    for AsyncPartialTmaJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, TmaTilingLayout>,
        barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);
        if this.is_elected {
            let size_row = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => config.smem_config.elements_per_stage_along_row(),
                MatrixLayout::ColMajor => config.smem_config.elements_per_stage_along_col(),
            };
            let size_col = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => config.smem_config.elements_per_tile_along_col,
                MatrixLayout::ColMajor => config.smem_config.elements_per_tile_along_row,
            };

            let (offs_row, offs_col) = comptime![match config.stage_ident {
                StageIdent::Lhs => (
                    0,
                    this.stage_index * config.smem_config.elements_per_stage_along_col()
                ),
                StageIdent::Rhs => (
                    this.stage_index * config.smem_config.elements_per_stage_along_row(),
                    0
                ),
                _ => (0, 0),
            }]
            .runtime();

            let global_view = global_iter.view();
            let mut stage = stage.as_slice_mut(1u32);
            let slice_size = size_row * size_col;

            let slice_start = task_id * slice_size;
            let slice = stage.slice_mut(slice_start, slice_start + slice_size);
            // "column" to be loaded, may be a row for col-major (can't think of a better name)
            let load_col = task_id * size_col;

            let pos = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => (offs_row, load_col + offs_col),
                MatrixLayout::ColMajor => (load_col + offs_row, offs_col),
            };

            global_view.tensor_map_load(barrier, &mut slice.try_cast_unchecked(), pos);
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}

#[cube]
impl AsyncPartialLoadingStrategy for AsyncPartialTmaLoading {
    fn arrival_count<S: StageConfig>(#[comptime] _config: SharedGlobalMatmulConfig<S>) -> u32 {
        1u32.runtime()
    }

    fn barrier_post_init() {
        sync_async_proxy_shared();
    }

    fn arrive<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    ) {
        let lhs_elem_size = LhsS::<MP>::type_size();
        let rhs_elem_size = RhsS::<MP>::type_size();
        let stage_bytes = comptime! {
            let lhs_bytes = config.lhs_reader_config().smem_config.elements_per_stage() * lhs_elem_size;
            let rhs_bytes = config.rhs_reader_config().smem_config.elements_per_stage() * rhs_elem_size;
            lhs_bytes + rhs_bytes
        };
        barrier.arrive_and_expect_tx(1, stage_bytes);
    }

    fn is_elected<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> bool {
        let role_rule = RoleRule::new(config.plane_role_config().rule);
        role_rule.elect_load_leader()
    }
}
