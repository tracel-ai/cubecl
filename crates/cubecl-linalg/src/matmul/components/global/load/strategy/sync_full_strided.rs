use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::global::load::strategy::base::default_sync_full_load;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, Quantization};
use crate::matmul::components::stage::{Stage, StridedTilingLayout};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoadingJob, LoadingJobConfig};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct LoadingStrategy {}

impl LoadingValidation for LoadingStrategy {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_lines = tiling.total_size() / line_size;
        let total_units = config.num_planes() * config.plane_dim();

        if num_stage_lines % total_units != 0 {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl SyncFullLoadingStrategy for LoadingStrategy {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = Job<MP>;

    fn load_full<MP: MatmulPrecision, G: GlobalConfig>(
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        default_sync_full_load::<Self, MP, G>(
            tensor_reader,
            stage,
            quantization,
            input_ident,
            config,
        )
    }

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tiling = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);
        let num_stage_lines = tiling.total_size() / line_size;
        let unit_count = config.num_planes() * config.plane_dim();
        let num_tasks = comptime!(num_stage_lines / unit_count);

        let unit_position_base = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        Job::<MP> {
            unit_position_base,
            quantization,
            job_config: comptime!(JobConfig {
                num_tasks,
                unit_count,
                line_size,
                input_ident
            }),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job<MP: MatmulPrecision> {
    unit_position_base: u32,

    quantization: CubeOption<Quantization<MP>>,

    #[cube(comptime)]
    job_config: JobConfig,
}

#[derive(Copy, Clone)]
pub struct JobConfig {
    num_tasks: u32,
    unit_count: u32,
    line_size: u32,
    input_ident: InputIdent,
}

impl<MP: MatmulPrecision> LoadingJobConfig<MP, StridedTilingLayout, Job<MP>> for JobConfig {
    fn len(job: &Job<MP>) -> u32 {
        job.job_config.num_tasks
    }

    fn __expand_len(
        _context: &mut cubecl_core::prelude::Scope,
        job: <Job<MP> as cubecl_core::prelude::CubeType>::ExpandType,
    ) -> u32 {
        job.job_config.num_tasks
    }
}

#[cube]
impl<MP: MatmulPrecision> LoadingJob<MP, StridedTilingLayout> for Job<MP> {
    type LoadingJobConfig = JobConfig;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, StridedTilingLayout>,
        #[comptime] config: G,
    ) {
        let jc = this.job_config;
        let unit_position = this.unit_position_base + task_id * jc.unit_count;

        let line_read = tensor_reader.load_coalesced_in_stage::<G>(
            unit_position * jc.line_size,
            jc.input_ident,
            config,
        );

        stage.as_slice_mut()[unit_position] = match this.quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read),
            CubeOption::None => Line::cast_from(line_read),
        }
    }
}
