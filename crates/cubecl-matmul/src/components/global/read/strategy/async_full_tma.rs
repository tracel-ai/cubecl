use crate::components::MatmulElems;
use crate::components::global::GlobalReaderConfig;
use crate::components::global::read::{validate_async_barrier, validate_tma};
use crate::components::global::{RoleRule, read::async_tma::AsyncTma};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{StridedStageMemory, SwizzleMode};
use crate::components::{InvalidConfigError, MatmulProblem};
use crate::components::{MatrixLayout, global::read::FullLoadingStrategy};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::components::{global::multi_stage::LoadMaxRoundPlaneCount, stage::TmaTilingLayout};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using TMA load instructions.
/// Uses special tiling to minimize the number of loads required. Issues one load for each
/// tile in the major dimension (i.e. `k` for col-major RHS).
pub struct AsyncFullTmaLoading {}

impl LoadingValidation for AsyncFullTmaLoading {
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

impl LoadMaxRoundPlaneCount for AsyncFullTmaLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        _tiles_per_stage: u32,
        _line_size: u8,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        // Not sure this is the best value, but TMA is executed per-warpgroup so this is the maximum
        // number of planes executing one set of TMA loads.
        4
    }
}

#[cube]
impl FullLoadingStrategy for AsyncFullTmaLoading {
    type TilingLayout = TmaTilingLayout;
    type SyncStrategy = AsyncTma;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullTmaJob;

    fn new_job<EG: Numeric, ES: Numeric>(
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

        AsyncFullTmaJob {
            is_elected,
            num_tasks,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullTmaJob {
    is_elected: bool,

    #[cube(comptime)]
    num_tasks: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, TmaTilingLayout, AsyncTma> for AsyncFullTmaJob {
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, TmaTilingLayout>,
        barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        if this.is_elected {
            let config = comptime![config.smem_config];

            let size_row = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_per_stage_along_row(),
                MatrixLayout::ColMajor => config.elements_per_stage_along_col(),
            };
            let size_col = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_per_tile_along_col,
                MatrixLayout::ColMajor => config.elements_per_tile_along_row,
            };

            let global_view = global_iter.view();
            let mut stage = stage.as_slice_mut(1u32);
            let slice_size = size_row * size_col;

            let slice_start = task_id * slice_size;
            let slice = stage.slice_mut(slice_start, slice_start + slice_size);
            let col = task_id * size_col;

            let pos = match config.matrix_layout {
                MatrixLayout::RowMajor => (0, col),
                MatrixLayout::ColMajor => (col, 0),
            };

            global_view.tensor_map_load(barrier, &mut slice.try_cast_unchecked(), pos);
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
