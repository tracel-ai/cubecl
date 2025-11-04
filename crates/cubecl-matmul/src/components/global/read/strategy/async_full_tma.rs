use crate::components::global::{GlobalConfig, read::async_tma::AsyncTma};
use crate::components::stage::StridedStage;
use crate::components::{InvalidConfigError, MatmulIdent};
use crate::components::{MatrixLayout, global::read::FullLoadingStrategy};
use crate::components::{MatrixPrecision, TilingScheme};
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
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        TmaTilingLayout::check(config.global_memory_config(ident))?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullTmaLoading {
    fn max_round_plane_count(
        _tiling_scheme: &TilingScheme,
        _ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
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
    type Job<IP: MatrixPrecision> = AsyncFullTmaJob;

    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] _line_size: u32,
        #[comptime] config: G,
    ) -> Self::Job<IP> {
        let config = config.stage_memory_config(ident);
        let tile_count_col = match config.matrix_layout {
            MatrixLayout::RowMajor => config.tiles_in_stage_col,
            MatrixLayout::ColMajor => config.tiles_in_stage_row,
        };

        AsyncFullTmaJob {
            is_elected: UNIT_POS == 0,
            num_tasks: tile_count_col,
            ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullTmaJob {
    is_elected: bool,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<IP: MatrixPrecision> LoadingJob<IP, TmaTilingLayout, AsyncTma> for AsyncFullTmaJob {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<IP::Global>>,
        stage: &mut StridedStage<IP::Stage, TmaTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: G,
    ) {
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
