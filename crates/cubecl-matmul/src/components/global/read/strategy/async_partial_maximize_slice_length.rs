use crate::components::{
    InvalidConfigError, MatmulElems, MatrixLayout, StageIdent,
    global::{
        GlobalReaderConfig,
        memory::{GlobalIterator, load_window_in_stage},
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            LoadingJob, PartialLoadingStrategy, async_barrier::AsyncBarrier,
            validate_async_barrier, validate_noswizzle,
        },
    },
    stage::{StridedStageFamily, StridedStageMemory, StridedTilingLayout, TilingValidation},
};
use cubecl_core::prelude::{barrier::Barrier, *};
use cubecl_core::{self as cubecl};

use super::LoadingValidation;

#[derive(CubeType, Clone, Copy)]
/// Executes one `memcpy_async` call per contiguous slice.
/// The goal is to reduce the total number of `memcpy_async` calls, though it may result in idle threads.
pub struct AsyncPartialMaximizeSliceLengthLoading {}

impl LoadingValidation for AsyncPartialMaximizeSliceLengthLoading {
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

impl LoadMaxRoundPlaneCount for AsyncPartialMaximizeSliceLengthLoading {
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
impl PartialLoadingStrategy for AsyncPartialMaximizeSliceLengthLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncBarrier;
    type Stage = StridedStageFamily;

    type Job<EG: Numeric, ES: Numeric> = AsyncPartialMaximizeSliceLengthJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncPartialMaximizeSliceLengthJob {
        let matrix_layout = config.gmem_config.matrix_layout;
        let num_stages = config.smem_config.num_stages;

        let total_row = config.smem_config.elements_per_stage_along_row();
        let total_col = config.smem_config.elements_per_stage_along_col();

        // If stage is parallel to slices, slices are as long as in full stage memory, but there are less.
        // Otherwise, slices are shorter but there are as many as in full stage memory
        let (num_slices, num_slices_stage_offset, slice_length, slice_stage_offset) = comptime! {
            match (config.stage_ident, matrix_layout) {
                (StageIdent::Lhs, MatrixLayout::RowMajor) => {
                    let slice_length = total_col / (num_stages * line_size);

                    (total_row, 0, slice_length, stage_index * slice_length)
                },
                (StageIdent::Lhs, MatrixLayout::ColMajor) => {
                    let num_slices = total_col / num_stages;

                    (num_slices, stage_index * num_slices, total_row / line_size, 0)
                },
                (StageIdent::Rhs, MatrixLayout::RowMajor) => {
                    let num_slices = total_row / num_stages;

                    (num_slices, stage_index * num_slices, total_col / line_size, 0)
                },
                (StageIdent::Rhs, MatrixLayout::ColMajor) => {
                    let slice_length = total_row / (num_stages * line_size);

                    (total_col, 0, slice_length, stage_index * slice_length)
                },
                (_, _) => unreachable!()
            }
        };

        let unit_count = config.loading_units_count();
        let num_tasks_per_unit = comptime!(num_slices.div_ceil(unit_count));

        AsyncPartialMaximizeSliceLengthJob {
            buffer_idx: stage_index,
            num_tasks_per_unit,
            unit_count,
            num_slices_stage_offset,
            slice_stage_offset,
            slice_length,
            num_slices,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialMaximizeSliceLengthJob {
    buffer_idx: u32,
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    num_slices_stage_offset: u32,
    #[cube(comptime)]
    slice_stage_offset: u32,
    #[cube(comptime)]
    slice_length: u32,
    #[cube(comptime)]
    num_slices: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncBarrier>
    for AsyncPartialMaximizeSliceLengthJob
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
        let mut stage = stage.with_buffer_index(this.buffer_idx);
        let nth_slice_in_stage = this.unit_count * task_id + UNIT_POS;

        let nth_slice = nth_slice_in_stage + this.num_slices_stage_offset;

        let window = load_window_in_stage(
            &global_iter.view(),
            nth_slice,
            config.smem_config,
            config.gmem_config,
        );
        let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES>(
            &mut stage,
            nth_slice_in_stage,
            comptime!(config.smem_config),
        );

        let start = this.slice_stage_offset;
        let limit = select(
            this.slice_stage_offset < window.len(),
            this.slice_stage_offset,
            window.len(),
        );
        let end = start + Min::min(window.len() - limit, this.slice_length);

        let src = window.slice(start, end);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_slices.is_multiple_of(this.unit_count)) {
            barrier.memcpy_async(&src.try_cast_unchecked(), &mut destination);
        } else {
            if nth_slice_in_stage < this.num_slices {
                barrier.memcpy_async(&src.try_cast_unchecked(), &mut destination);
            }
        };
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
