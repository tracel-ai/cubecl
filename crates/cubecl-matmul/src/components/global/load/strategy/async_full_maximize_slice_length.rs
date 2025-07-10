use crate::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig,
        load::AsyncFullLoadingStrategy,
        global_memory::{TensorReader, Window},
    },
    stage::{StageMemory, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};
use cubecl_std::div_ceil;

use super::{AsyncLoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per contiguous slice.
/// The goal is to reduce the total number of memcpy_async calls, though it may result in idle threads.
pub struct LoadingStrategy {}

impl LoadingValidation for LoadingStrategy {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
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

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.tiling_scheme().elements_in_stage_row(input_ident),
            MatrixLayout::ColMajor => config.tiling_scheme().elements_in_stage_col(input_ident),
        };
        let unit_count = config.plane_dim() * config.num_loading_planes(input_ident);

        let num_tasks_per_unit = comptime!(div_ceil(num_slices, unit_count));

        Job {
            num_tasks_per_unit,
            unit_count,
            num_slices,
            input_ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    num_slices: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision> AsyncLoadingJob<MP, StridedTilingLayout> for Job {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let nth_slice = this.unit_count * task_id + UNIT_POS;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_slices % this.unit_count == 0) {
            load_nth_slice::<MP::EI, MP::ES, CM, G>(
                nth_slice,
                tensor_reader,
                stage,
                mechanism,
                this.input_ident,
                config,
            );
        } else {
            if nth_slice < this.num_slices {
                load_nth_slice::<MP::EI, MP::ES, CM, G>(
                    nth_slice,
                    tensor_reader,
                    stage,
                    mechanism,
                    this.input_ident,
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
fn load_nth_slice<EG: Numeric, ES: Numeric, CM: CopyMechanism<ES>, G: GlobalConfig>(
    nth_slice: u32,
    tensor_reader: &TensorReader<EG>,
    stage: &mut StageMemory<ES, StridedTilingLayout>,
    mechanism: &CM,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let window: Window<EG> =
        tensor_reader.load_window_in_stage::<G>(nth_slice, input_ident, config);
    let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES, G::StageConfig>(
        stage,
        nth_slice,
        comptime!(input_ident.as_ident()),
        config.stage_config(),
    );

    CM::memcpy_async(
        mechanism,
        &window.slice.try_cast_unchecked(),
        &mut destination,
    );
}
