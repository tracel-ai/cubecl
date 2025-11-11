use crate::components::{
    InvalidConfigError, MatmulIdent, MatrixLayout, MatrixPrecision, TilingScheme,
    global::{
        GlobalConfig,
        memory::{GlobalIterator, load_window_in_stage},
        multi_stage::LoadMaxRoundPlaneCount,
        read::{FullLoadingStrategy, LoadingJob, async_barrier::AsyncBarrier},
    },
    stage::{StridedStageMemory, StridedStageFamily, StridedTilingLayout, TilingValidation},
};
use cubecl_core::prelude::{barrier::Barrier, *};
use cubecl_core::{self as cubecl};

use super::LoadingValidation;

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per contiguous slice.
/// The goal is to reduce the total number of memcpy_async calls, though it may result in idle threads.
pub struct AsyncFullMaximizeSliceLengthLoading {}

impl LoadingValidation for AsyncFullMaximizeSliceLengthLoading {
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        StridedTilingLayout::check(config.global_memory_config(ident))?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullMaximizeSliceLengthLoading {
    fn max_round_plane_count(
        _tiling_scheme: &TilingScheme,
        _ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
    ) -> u32 {
        // Not sure what's ideal here, the current specialization isn't great anyways so can deal
        // with it later
        4
    }
}

#[cube]
impl FullLoadingStrategy for AsyncFullMaximizeSliceLengthLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncBarrier;
    type Job<IP: MatrixPrecision> = AsyncFullMaximizeSliceLengthJob;

    const SHOULD_CLEAR: bool = true;

    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] _line_size: u32,
        #[comptime] config: G,
    ) -> AsyncFullMaximizeSliceLengthJob {
        let matrix_layout = config.matrix_layout(ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.tiling_scheme().elements_in_stage_row(ident),
            MatrixLayout::ColMajor => config.tiling_scheme().elements_in_stage_col(ident),
        };
        let unit_count = config.plane_dim() * config.num_loading_planes(ident);

        let num_tasks_per_unit = comptime!(div_ceil(num_slices, unit_count));

        AsyncFullMaximizeSliceLengthJob {
            num_tasks_per_unit,
            unit_count,
            num_slices,
            ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullMaximizeSliceLengthJob {
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    num_slices: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<IP: MatrixPrecision> LoadingJob<IP, StridedTilingLayout, AsyncBarrier>
    for AsyncFullMaximizeSliceLengthJob
{
    type Stage = StridedStageFamily;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<IP::Global>>,
        stage: &mut StridedStageMemory<IP::Stage, StridedTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: G,
    ) {
        let nth_slice = this.unit_count * task_id + UNIT_POS;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_slices.is_multiple_of(this.unit_count)) {
            load_nth_slice::<IP::Global, IP::Stage, G>(
                nth_slice,
                global_iter,
                stage,
                barrier,
                this.ident,
                config,
            );
        } else {
            if nth_slice < this.num_slices {
                load_nth_slice::<IP::Global, IP::Stage, G>(
                    nth_slice,
                    global_iter,
                    stage,
                    barrier,
                    this.ident,
                    config,
                );
            }
        };
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
fn load_nth_slice<EG: Numeric, ES: Numeric, G: GlobalConfig>(
    nth_slice: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut StridedStageMemory<ES, StridedTilingLayout>,
    barrier: &Barrier,
    #[comptime] ident: MatmulIdent,
    #[comptime] config: G,
) {
    let window = load_window_in_stage(
        &global_iter.view(),
        nth_slice,
        comptime!(config.global_memory_config(ident)),
    );
    let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES>(
        stage,
        nth_slice,
        comptime!(config.stage_memory_config(ident)),
    );

    barrier.memcpy_async(&window.try_cast_unchecked(), &mut destination);
}
