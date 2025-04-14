use std::marker::PhantomData;

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, Stage, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoadingJob, LoadingJobConfig};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for LoadingStrategy<TO> {
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
impl<TO: TilingOrder> SyncFullLoadingStrategy for LoadingStrategy<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<MP: MatmulPrecision> = Job<MP>;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tiling = config.tiling_dimensions(input_ident);
        let tile_num_elements = tiling.tile_size();
        let line_size = config.global_line_size(input_ident);
        let num_stage_elements = tiling.total_size();
        let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
        let num_tasks = comptime!(num_stage_elements / jump_length);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        Job::<MP> {
            unit_position_base,
            quantization,
            job_config: comptime!(JobConfig {
                num_tasks,
                tile_num_elements,
                jump_length,
                line_size,
                input_ident,
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
    tile_num_elements: u32,
    jump_length: u32,
    line_size: u32,
    input_ident: InputIdent,
}

impl<MP: MatmulPrecision, T: TilingOrder> LoadingJobConfig<MP, ContiguousTilingLayout<T>, Job<MP>>
    for JobConfig
{
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
impl<MP: MatmulPrecision, TO: TilingOrder> LoadingJob<MP, ContiguousTilingLayout<TO>> for Job<MP> {
    type LoadingJobConfig = JobConfig;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, ContiguousTilingLayout<TO>>,
        #[comptime] config: G,
    ) {
        let jc = this.job_config;

        let unit_position = this.unit_position_base + task_id * jc.jump_length;

        let nth_tile = unit_position / jc.tile_num_elements;
        let pos_within_tile = unit_position % jc.tile_num_elements;

        let (tile_x, tile_y) = ContiguousTilingLayout::<TO>::to_x_y::<G::SmmConfig>(
            nth_tile,
            comptime!(jc.input_ident.as_ident()),
            comptime!(config.to_smm_config()),
        );

        let line_read = tensor_reader.load_coalesced_in_tile::<G>(
            tile_x,
            tile_y,
            pos_within_tile,
            jc.input_ident,
            config,
        );

        stage.as_slice_mut()[unit_position / jc.line_size] = match this.quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read, jc.input_ident),
            CubeOption::None => Line::cast_from(line_read),
        };
    }
}
