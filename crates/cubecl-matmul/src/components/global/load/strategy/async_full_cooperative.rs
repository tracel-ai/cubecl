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

use super::{AsyncLoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// Each `memcpy_async` is called with the same arguments for cooperative behaviour
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

        Job {
            num_slices,
            input_ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
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
        let window: Window<MP::EI> =
            tensor_reader.load_window_in_stage::<G>(task_id, this.input_ident, config);
        let mut destination: SliceMut<Line<MP::ES>> =
            StridedTilingLayout::nth_slice::<MP::ES, G::StageConfig>(
                stage,
                task_id,
                comptime!(this.input_ident.as_ident()),
                config.stage_config(),
            );

        CM::memcpy_async(
            mechanism,
            &window.slice.try_cast_unchecked(),
            &mut destination,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_slices
    }
}
