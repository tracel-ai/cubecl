use crate::components::{
    InvalidConfigError, MatmulIdent, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig,
        global_memory::{TensorReader, Window},
        load::AsyncFullLoadingStrategy,
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
pub struct AsyncFullMaximizeSliceLengthLoading {}

impl LoadingValidation for AsyncFullMaximizeSliceLengthLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncFullLoadingStrategy for AsyncFullMaximizeSliceLengthLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = AsynFullMaximizeSliceLengthJob;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> AsynFullMaximizeSliceLengthJob {
        let matrix_layout = config.matrix_layout(ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.tiling_scheme().elements_in_stage_row(ident),
            MatrixLayout::ColMajor => config.tiling_scheme().elements_in_stage_col(ident),
        };
        let unit_count = config.plane_dim() * config.num_loading_planes(ident);

        let num_tasks_per_unit = comptime!(div_ceil(num_slices, unit_count));

        AsynFullMaximizeSliceLengthJob {
            num_tasks_per_unit,
            unit_count,
            num_slices,
            ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsynFullMaximizeSliceLengthJob {
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
impl<MP: MatmulPrecision> AsyncLoadingJob<MP, StridedTilingLayout>
    for AsynFullMaximizeSliceLengthJob
{
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
                this.ident,
                config,
            );
        } else {
            if nth_slice < this.num_slices {
                load_nth_slice::<MP::EI, MP::ES, CM, G>(
                    nth_slice,
                    tensor_reader,
                    stage,
                    mechanism,
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
fn load_nth_slice<EG: Numeric, ES: Numeric, CM: CopyMechanism<ES>, G: GlobalConfig>(
    nth_slice: u32,
    tensor_reader: &TensorReader<EG>,
    stage: &mut StageMemory<ES, StridedTilingLayout>,
    mechanism: &CM,
    #[comptime] ident: MatmulIdent,
    #[comptime] config: G,
) {
    let window: Window<EG> = tensor_reader.load_window_in_stage::<G>(nth_slice, ident, config);
    let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES, G::StageConfig>(
        stage,
        nth_slice,
        comptime!(ident.into_stage()),
        config.stage_config(),
    );

    CM::memcpy_async(
        mechanism,
        &window.slice.try_cast_unchecked(),
        &mut destination,
    );
}
