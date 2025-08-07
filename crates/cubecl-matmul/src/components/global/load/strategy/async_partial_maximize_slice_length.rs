use crate::components::{
    InputPrecision, InvalidConfigError, MatmulIdent, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig,
        load::AsyncPartialLoadingStrategy,
        memory::{TensorReader, Window},
    },
    stage::{StageConfig, StageMemory, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::{AsyncLoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Executes one `memcpy_async` call per contiguous slice.
/// The goal is to reduce the total number of `memcpy_async` calls, though it may result in idle threads.
pub struct AsyncPartialMaximizeSliceLengthLoading {}

impl LoadingValidation for AsyncPartialMaximizeSliceLengthLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncPartialLoadingStrategy for AsyncPartialMaximizeSliceLengthLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<IP: InputPrecision> = AsyncPartialMaximizeSliceLengthJob;

    fn new_job<IP: InputPrecision, G: GlobalConfig>(
        #[comptime] stage_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> AsyncPartialMaximizeSliceLengthJob {
        let matrix_layout = config.matrix_layout(ident);
        let line_size = config
            .stage_config()
            .stage_line_size(comptime!(ident.into_stage()));
        let num_stages = 2;

        let total_row = config.tiling_scheme().elements_in_stage_row(ident);
        let total_col = config.tiling_scheme().elements_in_stage_col(ident);

        // If stage is parallel to slices, slices are as long as in full stage memory, but there are less.
        // Otherwise, slices are shorter but there are as many as in full stage memory
        let (num_slices, num_slices_stage_offset, slice_length, slice_stage_offset) = comptime! {
            match (ident, matrix_layout) {
                (MatmulIdent::Lhs, MatrixLayout::RowMajor) => {
                    let slice_length = total_col / (num_stages * line_size);

                    (total_row, 0, slice_length, stage_index * slice_length)
                },
                (MatmulIdent::Lhs, MatrixLayout::ColMajor) => {
                    let num_slices = total_col / num_stages;

                    (num_slices, stage_index * num_slices, total_row / line_size, 0)
                },
                (MatmulIdent::Rhs, MatrixLayout::RowMajor) => {
                    let num_slices = total_row / num_stages;

                    (num_slices, stage_index * num_slices, total_col / line_size, 0)
                },
                (MatmulIdent::Rhs, MatrixLayout::ColMajor) => {
                    let slice_length = total_row / (num_stages * line_size);

                    (total_col, 0, slice_length, stage_index * slice_length)
                },
                (MatmulIdent::Out, _) => unreachable!()
            }
        };

        let unit_count = config.plane_dim() * config.num_loading_planes(ident);
        let num_tasks_per_unit = comptime!(num_slices.div_ceil(unit_count));

        AsyncPartialMaximizeSliceLengthJob {
            num_tasks_per_unit,
            unit_count,
            num_slices_stage_offset,
            ident,
            slice_stage_offset,
            slice_length,
            num_slices,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialMaximizeSliceLengthJob {
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    num_slices_stage_offset: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    slice_stage_offset: u32,
    #[cube(comptime)]
    slice_length: u32,
    #[cube(comptime)]
    num_slices: u32,
}

#[cube]
impl<IP: InputPrecision> AsyncLoadingJob<IP, StridedTilingLayout>
    for AsyncPartialMaximizeSliceLengthJob
{
    fn execute_task<CM: CopyMechanism<IP::Stage>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<IP::Global>,
        stage: &mut StageMemory<IP::Stage, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let nth_slice_in_stage = this.unit_count * task_id + UNIT_POS;

        let nth_slice = nth_slice_in_stage + this.num_slices_stage_offset;

        let window: Window<IP::Global> = tensor_reader.load_window_in_stage(
            nth_slice,
            comptime!(config.global_memory_config(this.ident)),
        );
        let mut destination: SliceMut<Line<IP::Stage>> =
            StridedTilingLayout::nth_slice::<IP::Stage, G::StageMemoryConfig>(
                stage,
                nth_slice,
                comptime!(this.ident.into_stage()),
                config.stage_memory_config(),
            );

        let start = this.slice_stage_offset;
        let limit = select(
            this.slice_stage_offset < window.size,
            this.slice_stage_offset,
            window.size,
        );
        let end = start + Min::min(window.size - limit, this.slice_length);

        let src = window.slice.slice(start, end);
        let mut dest = destination.slice_mut(start, end);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_slices % this.unit_count == 0) {
            CM::memcpy_async(mechanism, &src.try_cast_unchecked(), &mut dest);
        } else {
            if nth_slice_in_stage < this.num_slices {
                CM::memcpy_async(mechanism, &src.try_cast_unchecked(), &mut dest);
            }
        };
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
