use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation, Quantization,
        load::AsyncFullLoadingStrategy,
        tensor_view::{TensorReader, Window},
    },
    stage::{Stage, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};
use cubecl_std::CubeOption;

use super::{AsyncLoadingJob, AsyncLoadingJobConfig};

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
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job {
        comptime! {
            if quantization.is_some() {
                panic!("Quantization not supported on async loaders.")
            }
        }

        let matrix_layout = config.matrix_layout(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => tiling_dimensions.total_row(),
            MatrixLayout::ColMajor => tiling_dimensions.total_col(),
        };

        Job {
            job_config: comptime!(JobConfig {
                num_slices,
                input_ident,
            }),
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    #[cube(comptime)]
    job_config: JobConfig,
}

#[derive(Clone, Copy)]
pub struct JobConfig {
    num_slices: u32,
    input_ident: InputIdent,
}

impl<MP: MatmulPrecision> AsyncLoadingJobConfig<MP, StridedTilingLayout, Job> for JobConfig {
    fn len(job: &Job) -> u32 {
        job.job_config.num_slices
    }

    fn __expand_len(
        _context: &mut cubecl_core::prelude::Scope,
        job: <Job as cubecl_core::prelude::CubeType>::ExpandType,
    ) -> u32 {
        job.job_config.num_slices
    }
}

#[cube]
impl<MP: MatmulPrecision> AsyncLoadingJob<MP, StridedTilingLayout> for Job {
    type LoadingJobConfig = JobConfig;

    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let input_ident = comptime!(this.job_config.input_ident);

        let window: Window<MP::EI> =
            tensor_reader.load_window_in_stage::<G>(task_id, input_ident, config);
        let mut destination: SliceMut<Line<MP::ES>> =
            StridedTilingLayout::nth_slice::<MP::ES, G::SmmConfig>(
                stage,
                task_id,
                comptime!(input_ident.as_ident()),
                config.to_smm_config(),
            );

        CM::memcpy_async(
            mechanism,
            &window.slice.try_cast_unchecked(),
            &mut destination,
        );
    }
}
