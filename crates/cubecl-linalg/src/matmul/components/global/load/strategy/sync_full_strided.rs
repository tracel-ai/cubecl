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
pub struct SyncFullStridedLoading {}

impl LoadingValidation for SyncFullStridedLoading {
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
impl SyncFullLoadingStrategy for SyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = SyncFullStridedJob<MP>;

    fn load_full<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        default_sync_full_load::<Self, MP, G>(read_view, stage, quantization, input_ident, config)
    }

    fn job<MP: MatmulPrecision, G: GlobalConfig>(
        stage: Stage<MP::ES, Self::TilingLayout>,
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

        SyncFullStridedJob::<MP> {
            unit_position_base,
            stage,
            quantization,
            job_config: comptime!(SyncFullStridedJobConfig {
                num_tasks,
                unit_count,
                line_size,
                input_ident
            }),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncFullStridedJob<MP: MatmulPrecision> {
    unit_position_base: u32,

    stage: Stage<MP::ES, StridedTilingLayout>,
    quantization: CubeOption<Quantization<MP>>,

    #[cube(comptime)]
    job_config: SyncFullStridedJobConfig,
}

#[derive(Copy, Clone)]
pub struct SyncFullStridedJobConfig {
    num_tasks: u32,
    unit_count: u32,
    line_size: u32,
    input_ident: InputIdent,
}

impl<MP: MatmulPrecision> LoadingJobConfig<MP, SyncFullStridedJob<MP>>
    for SyncFullStridedJobConfig
{
    fn len(job: &SyncFullStridedJob<MP>) -> u32 {
        job.job_config.num_tasks
    }

    fn __expand_len(
        _context: &mut cubecl_core::prelude::Scope,
        job: <SyncFullStridedJob<MP> as cubecl_core::prelude::CubeType>::ExpandType,
    ) -> u32 {
        job.job_config.num_tasks
    }
}

#[cube]
impl<MP: MatmulPrecision> LoadingJob<MP> for SyncFullStridedJob<MP> {
    type LoadingJobConfig = SyncFullStridedJobConfig;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        read_view: &TensorReader<MP::EI>,
        #[comptime] config: G,
    ) {
        let jc = this.job_config;
        let unit_position = this.unit_position_base + task_id * jc.unit_count;

        let line_read = read_view.load_coalesced_in_stage::<G>(
            unit_position * jc.line_size,
            jc.input_ident,
            config,
        );

        this.stage.as_slice_mut()[unit_position] = match this.quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read),
            CubeOption::None => Line::cast_from(line_read),
        }
    }
}
