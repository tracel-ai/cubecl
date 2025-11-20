use crate::components::MatmulElems;
use crate::components::global::GlobalReaderConfig;
use crate::components::global::read::{validate_async_barrier, validate_tma};
use crate::components::global::{RoleRule, multi_stage::LoadMaxRoundPlaneCount};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::TmaTilingLayout;
use crate::components::stage::{StridedStageMemory, SwizzleMode};
use crate::components::{InvalidConfigError, MatmulIdent, StageIdent, TilingScheme};
use crate::components::{
    MatrixLayout,
    global::read::{PartialLoadingStrategy, async_tma::AsyncTma},
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
        client: &ComputeClient<R::Server>,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        TmaTilingLayout::check(config.smem_config)?;
        validate_tma::<R>(client, config.smem_config, config.stage_ident, dtypes)?;

        validate_async_barrier::<R>(client)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialTmaLoading {
    fn max_round_plane_count(
        _tiling_scheme: &TilingScheme,
        _ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
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
            MatrixLayout::RowMajor => config.tiles_in_stage_col(),
            MatrixLayout::ColMajor => config.tiles_in_stage_row(),
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
        barrier: &mut Barrier,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);
        if this.is_elected {
            let size_row = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => config.smem_config.elements_in_stage_row(),
                MatrixLayout::ColMajor => config.smem_config.elements_in_stage_col(),
            };
            let size_col = match config.smem_config.matrix_layout {
                MatrixLayout::RowMajor => config.smem_config.elements_per_tile_col,
                MatrixLayout::ColMajor => config.smem_config.elements_per_tile_row,
            };

            let (offs_row, offs_col) = comptime![match config.stage_ident {
                StageIdent::Lhs => (
                    0,
                    this.stage_index * config.smem_config.elements_in_stage_col()
                ),
                StageIdent::Rhs => (
                    this.stage_index * config.smem_config.elements_in_stage_row(),
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
