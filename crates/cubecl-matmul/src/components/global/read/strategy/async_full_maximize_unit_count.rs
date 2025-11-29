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
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct AsyncFullMaximizeUnitCountLoading {}

impl LoadingValidation for AsyncFullMaximizeUnitCountLoading {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        config: &GlobalReaderConfig,
        _dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        let matrix_layout = config.gmem_config.matrix_layout;
        let line_size = config.gmem_config.line_size;

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                config.smem_config.elements_per_stage_along_row(),
                config.smem_config.elements_per_stage_along_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.smem_config.elements_per_stage_along_col(),
                config.smem_config.elements_per_stage_along_row() / line_size,
            ),
        };
        let unit_count = config.plane_dim * config.loading_planes_count();

        if !unit_count.is_multiple_of(num_slices) {
            return Err(Box::new(
                "Number of slices must divide number of units evenly",
            ));
        }
        if slice_length % (unit_count / num_slices) != 0 {
            return Err(Box::new(
                "Number of units per slice must divide slice length evenly",
            ));
        }

        StridedTilingLayout::check(config.smem_config)?;
        validate_async_barrier(client)?;
        validate_noswizzle(config.smem_config)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullMaximizeUnitCountLoading {
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
impl FullLoadingStrategy for AsyncFullMaximizeUnitCountLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncBarrier;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullMaximizeUnitCountJob;

    const SHOULD_CLEAR: bool = true;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncFullMaximizeUnitCountJob {
        let matrix_layout = config.gmem_config.matrix_layout;

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                config.smem_config.elements_per_stage_along_row(),
                config.smem_config.elements_per_stage_along_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.smem_config.elements_per_stage_along_col(),
                config.smem_config.elements_per_stage_along_row() / line_size,
            ),
        };

        let unit_count = config.loading_units_count();

        let units_per_slice = comptime!(unit_count / num_slices);
        let nth_slice = UNIT_POS / units_per_slice;

        let segment_length = comptime!(slice_length / units_per_slice);
        let nth_segment = UNIT_POS % units_per_slice;

        AsyncFullMaximizeUnitCountJob {
            nth_slice,
            nth_segment,
            segment_length,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullMaximizeUnitCountJob {
    nth_slice: u32,
    nth_segment: u32,
    #[cube(comptime)]
    segment_length: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncBarrier>
    for AsyncFullMaximizeUnitCountJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] _task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, StridedTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES>(
            stage,
            this.nth_slice,
            comptime!(config.smem_config),
        );

        let window = load_window_in_stage(
            &global_iter.view(),
            this.nth_slice,
            config.smem_config,
            config.gmem_config,
        );
        let seg_start = Min::min(this.nth_segment * this.segment_length, window.len());
        let seg_end = Min::min((this.nth_segment + 1) * this.segment_length, window.len());

        let src_segment = window.slice(seg_start, seg_end);
        let mut dest_segment = destination.slice_mut(seg_start, seg_end);

        barrier.memcpy_async(&src_segment.try_cast_unchecked(), &mut dest_segment);
    }

    fn task_count(_this: &Self) -> comptime_type!(u32) {
        1
    }
}
