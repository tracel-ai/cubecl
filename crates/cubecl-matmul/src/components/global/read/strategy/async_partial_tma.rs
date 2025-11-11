use crate::components::global::read::{validate_async_barrier, validate_tma};
use crate::components::global::{RoleRule, multi_stage::LoadMaxRoundPlaneCount};
use crate::components::stage::StridedStage;
use crate::components::{InvalidConfigError, MatmulIdent, TilingScheme};
use crate::components::{
    MatrixLayout,
    global::read::{PartialLoadingStrategy, async_tma::AsyncTma},
};
use crate::components::{global::GlobalConfig, stage::TmaTilingLayout};
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
    fn check<C: GlobalConfig, R: Runtime>(
        client: &ComputeClient<R::Server>,
        config: &C,
        ident: MatmulIdent,
    ) -> Result<(), InvalidConfigError> {
        TmaTilingLayout::check(config.global_memory_config(ident))?;
        validate_tma::<R>(client)?;
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
    type Job<EG: Numeric, ES: Numeric> = AsyncPartialTmaJob;

    fn new_job<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        #[comptime] stage_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] _line_size: u32,
        #[comptime] config: G,
    ) -> Self::Job<EG, ES> {
        let role_rule_config = config.role_rule_config();
        let config = config.stage_memory_config(ident);
        let tile_count_col = match config.matrix_layout {
            MatrixLayout::RowMajor => config.tiles_in_stage_col,
            MatrixLayout::ColMajor => config.tiles_in_stage_row,
        };

        let is_elected = RoleRule::new(role_rule_config).elect_load_leader();

        AsyncPartialTmaJob {
            is_elected,
            num_tasks: tile_count_col,
            stage_index,
            ident,
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
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, TmaTilingLayout, AsyncTma>
    for AsyncPartialTmaJob
{
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, TmaTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: G,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);
        if this.is_elected {
            let config = comptime![config.stage_memory_config(this.ident)];

            let size_row = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_in_stage_row(),
                MatrixLayout::ColMajor => config.elements_in_stage_col(),
            };
            let size_col = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_in_tile_col,
                MatrixLayout::ColMajor => config.elements_in_tile_row,
            };

            let (offs_row, offs_col) = comptime![match this.ident {
                MatmulIdent::Lhs => (0, this.stage_index * config.elements_in_stage_col()),
                MatmulIdent::Rhs => (this.stage_index * config.elements_in_stage_row(), 0),
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

            let pos = match config.matrix_layout {
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
