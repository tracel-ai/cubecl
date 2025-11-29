use crate::components::{
    InvalidConfigError, MatmulElems, MatrixLayout,
    global::{
        GlobalReaderConfig,
        memory::{GlobalIterator, load_window_in_stage},
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            FullLoadingStrategy, LoadingJob, async_barrier::AsyncBarrier, validate_async_barrier,
            validate_noswizzle,
        },
    },
    stage::{StridedStageFamily, StridedStageMemory, StridedTilingLayout, TilingValidation},
};
use cubecl_core::prelude::{barrier::Barrier, *};
use cubecl_core::{self as cubecl};

use super::LoadingValidation;

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per contiguous slice.
/// The goal is to reduce the total number of memcpy_async calls, though it may result in idle threads.
pub struct AsyncFullMaximizeSliceLengthLoading {}

impl LoadingValidation for AsyncFullMaximizeSliceLengthLoading {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        config: &GlobalReaderConfig,
        _dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        StridedTilingLayout::check(config.smem_config)?;
        validate_async_barrier(client)?;
        validate_noswizzle(config.smem_config)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullMaximizeSliceLengthLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        _tiles_per_stage: u32,
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
    type Job<EG: Numeric, ES: Numeric> = AsyncFullMaximizeSliceLengthJob;

    const SHOULD_CLEAR: bool = true;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] _line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncFullMaximizeSliceLengthJob {
        let matrix_layout = config.gmem_config.matrix_layout;

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.smem_config.elements_per_stage_along_row(),
            MatrixLayout::ColMajor => config.smem_config.elements_per_stage_along_col(),
        };
        let unit_count = config.loading_units_count();

        let num_tasks_per_unit = comptime!(div_ceil(num_slices, unit_count));

        AsyncFullMaximizeSliceLengthJob {
            num_tasks_per_unit,
            unit_count,
            num_slices,
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
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncBarrier>
    for AsyncFullMaximizeSliceLengthJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, StridedTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let nth_slice = this.unit_count * task_id + UNIT_POS;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_slices.is_multiple_of(this.unit_count)) {
            load_nth_slice::<EG, ES>(nth_slice, global_iter, stage, barrier, config);
        } else {
            if nth_slice < this.num_slices {
                load_nth_slice::<EG, ES>(nth_slice, global_iter, stage, barrier, config);
            }
        };
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
fn load_nth_slice<EG: Numeric, ES: Numeric>(
    nth_slice: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut StridedStageMemory<ES, StridedTilingLayout>,
    barrier: &Barrier,
    #[comptime] config: GlobalReaderConfig,
) {
    let window = load_window_in_stage(
        &global_iter.view(),
        nth_slice,
        config.smem_config,
        config.gmem_config,
    );
    let mut destination: SliceMut<Line<ES>> =
        StridedTilingLayout::nth_slice::<ES>(stage, nth_slice, comptime!(config.smem_config));

    barrier.memcpy_async(&window.try_cast_unchecked(), &mut destination);
}
