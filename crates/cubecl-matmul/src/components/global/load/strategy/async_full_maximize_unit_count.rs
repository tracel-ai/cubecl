use crate::components::{
    InputPrecision, InvalidConfigError, MatmulIdent, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig,
        load::AsyncFullLoadingStrategy,
        memory::{TensorReader, Window},
    },
    stage::{StageConfig, StageMemory, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::{AsyncLoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct AsyncFullMaximizeUnitCountLoading {}

impl LoadingValidation for AsyncFullMaximizeUnitCountLoading {
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        let matrix_layout = config.matrix_layout(ident);
        let line_size = config.global_line_size(ident);

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                config.tiling_scheme().elements_in_stage_row(ident),
                config.tiling_scheme().elements_in_stage_col(ident) / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.tiling_scheme().elements_in_stage_col(ident),
                config.tiling_scheme().elements_in_stage_row(ident) / line_size,
            ),
        };
        let unit_count = config.plane_dim() * config.num_loading_planes(ident);

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

        Ok(())
    }
}

#[cube]
impl AsyncFullLoadingStrategy for AsyncFullMaximizeUnitCountLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<IP: InputPrecision> = AsyncFullMaximizeUnitCountJob;

    fn new_job<IP: InputPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> AsyncFullMaximizeUnitCountJob {
        let matrix_layout = config.matrix_layout(ident);
        let line_size = config
            .stage_config()
            .stage_line_size(comptime!(ident.into_stage()));

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                config.tiling_scheme().elements_in_stage_row(ident),
                config.tiling_scheme().elements_in_stage_col(ident) / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.tiling_scheme().elements_in_stage_col(ident),
                config.tiling_scheme().elements_in_stage_row(ident) / line_size,
            ),
        };

        let unit_count = config.plane_dim() * config.num_loading_planes(ident);

        let units_per_slice = comptime!(unit_count / num_slices);
        let nth_slice = UNIT_POS / units_per_slice;

        let segment_length = comptime!(slice_length / units_per_slice);
        let nth_segment = UNIT_POS % units_per_slice;

        AsyncFullMaximizeUnitCountJob {
            nth_slice,
            nth_segment,
            segment_length,
            ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullMaximizeUnitCountJob {
    nth_slice: u32,
    nth_segment: u32,
    #[cube(comptime)]
    segment_length: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<IP: InputPrecision> AsyncLoadingJob<IP, StridedTilingLayout>
    for AsyncFullMaximizeUnitCountJob
{
    fn execute_task<CM: CopyMechanism, G: GlobalConfig>(
        this: &mut Self,
        _task_id: u32,
        tensor_reader: &TensorReader<IP::Global>,
        stage: &mut StageMemory<IP::Stage, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let mut destination: SliceMut<Line<IP::Stage>> =
            StridedTilingLayout::nth_slice::<IP::Stage, G::StageMemoryConfig>(
                stage,
                this.nth_slice,
                comptime!(this.ident.into_stage()),
                config.stage_memory_config(),
            );

        let window: Window<IP::Global> = tensor_reader.load_window_in_stage(
            this.nth_slice,
            comptime!(config.global_memory_config(this.ident)),
        );
        let seg_start = Min::min(this.nth_segment * this.segment_length, window.size);
        let seg_end = Min::min((this.nth_segment + 1) * this.segment_length, window.size);

        let src_segment = window.slice.slice(seg_start, seg_end);
        let mut dest_segment = destination.slice_mut(seg_start, seg_end);

        CM::memcpy_async(
            mechanism,
            &src_segment.try_cast_unchecked(),
            &mut dest_segment,
        );
    }

    fn task_count(_this: &Self) -> comptime_type!(u32) {
        1
    }
}
