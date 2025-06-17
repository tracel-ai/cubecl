use crate::components::global::load::SyncFullLoadingStrategy;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::tensor_view::TensorReader;
use crate::components::global::{GlobalConfig, Quantization, RoleRule};
use crate::components::stage::{StageMemory, StridedTilingLayout};
use crate::components::{Ident, InputIdent, InvalidConfigError};
use crate::components::{MatmulPrecision, TilingScheme};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct LoadingStrategy {}

impl LoadingValidation for LoadingStrategy {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let line_size = config.global_line_size(ident);

        let num_stage_lines = config.tiling_scheme().elements_in_stage(ident) / line_size;
        let total_units = config.num_loading_planes(ident) * config.plane_dim();

        if num_stage_lines % total_units != 0 {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for LoadingStrategy {
    fn max_round_plane_count(
        tiling_scheme: &TilingScheme,
        ident: InputIdent,
        line_size: u8,
        plane_dim: u32,
    ) -> u32 {
        let num_lines = tiling_scheme.elements_in_stage(ident) / line_size as u32;
        num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl SyncFullLoadingStrategy for LoadingStrategy {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let line_size = config.global_line_size(input_ident);
        let num_stage_lines = config.tiling_scheme().elements_in_stage(input_ident) / line_size;
        let unit_count = config.num_loading_planes(input_ident) * config.plane_dim();
        let num_tasks_per_unit = comptime!(num_stage_lines / unit_count);

        let unit_position_base = RoleRule::new(config.role_rule_config())
            .load_index(input_ident, config.specialized_loading_sides())
            * config.plane_dim()
            + UNIT_POS_X;

        Job {
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            line_size,
            input_ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision> LoadingJob<MP, StridedTilingLayout> for Job {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, StridedTilingLayout>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;

        let line_read = tensor_reader.load_coalesced_in_stage::<G>(
            unit_position * this.line_size,
            this.input_ident,
            config,
        );

        stage.as_slice_mut(this.line_size)[unit_position] = match quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read, this.input_ident),
            CubeOption::None => Line::cast_from(line_read),
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
