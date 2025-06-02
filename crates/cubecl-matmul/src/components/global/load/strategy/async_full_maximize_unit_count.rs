use crate::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig,
        load::AsyncFullLoadingStrategy,
        tensor_view::{TensorReader, Window},
    },
    stage::{StageConfig, StageMemory, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::{AsyncLoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct LoadingStrategy {}

impl LoadingValidation for LoadingStrategy {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
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
        let unit_count = config.plane_dim() * config.num_planes();

        if unit_count % num_slices != 0 {
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
impl AsyncFullLoadingStrategy for LoadingStrategy {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job {
        let matrix_layout = config.matrix_layout(input_ident);
        let line_size = config.stage_config().stage_line_size(input_ident.into());

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                config.tiling_scheme().elements_in_stage_row(input_ident),
                config.tiling_scheme().elements_in_stage_col(input_ident) / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.tiling_scheme().elements_in_stage_col(input_ident),
                config.tiling_scheme().elements_in_stage_row(input_ident) / line_size,
            ),
        };

        let unit_count = config.plane_dim() * config.num_planes();

        let units_per_slice = comptime!(unit_count / num_slices);
        let nth_slice = UNIT_POS / units_per_slice;

        let segment_length = comptime!(slice_length / units_per_slice);
        let nth_segment = UNIT_POS % units_per_slice;

        Job {
            nth_slice,
            nth_segment,
            segment_length,
            input_ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    nth_slice: u32,
    nth_segment: u32,
    #[cube(comptime)]
    segment_length: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision> AsyncLoadingJob<MP, StridedTilingLayout> for Job {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        _task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let mut destination: SliceMut<Line<MP::ES>> =
            StridedTilingLayout::nth_slice::<MP::ES, G::StageConfig>(
                stage,
                this.nth_slice,
                comptime!(this.input_ident.as_ident()),
                config.stage_config(),
            );

        let window: Window<MP::EI> =
            tensor_reader.load_window_in_stage::<G>(this.nth_slice, this.input_ident, config);
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
