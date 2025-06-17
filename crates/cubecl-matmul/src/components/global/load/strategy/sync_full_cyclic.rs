use std::marker::PhantomData;

use crate::components::global::load::SyncFullLoadingStrategy;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::tensor_view::TensorReader;
use crate::components::global::{GlobalConfig, Quantization, RoleRule};
use crate::components::stage::{ContiguousTilingLayout, StageMemory, TilingOrder};
use crate::components::{Ident, InputIdent, InvalidConfigError};
use crate::components::{MatmulPrecision, TilingScheme};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoaderMode, LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for LoadingStrategy<TO> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        if let LoaderMode::Strict = config.loader_mode() {
            let line_size = config.global_line_size(ident);

            let num_stage_lines = config.tiling_scheme().elements_in_stage(ident) / line_size;
            let total_units = config.num_loading_planes(ident) * config.plane_dim();

            if num_stage_lines % total_units != 0 {
                return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
            }
        }

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for LoadingStrategy<TO> {
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
impl<TO: TilingOrder> SyncFullLoadingStrategy for LoadingStrategy<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tile_num_elements = config.tiling_scheme().elements_in_tile(input_ident);
        let line_size = config.global_line_size(input_ident);
        let num_stage_elements = config.tiling_scheme().elements_in_stage(input_ident);
        let total_units = comptime!(config.num_loading_planes(input_ident) * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_tasks_per_unit = comptime!(num_stage_elements.div_ceil(jump_length));
        let balanced_workload = num_tasks_per_unit % total_units == 0;

        let unit_id = RoleRule::new(config.role_rule_config())
            .load_index(input_ident, config.specialized_loading_sides())
            * config.plane_dim()
            + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        Job {
            unit_position_base,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            line_size,
            input_ident,
            balanced_workload,
            num_stage_elements,
            loader_mode: comptime!(config.loader_mode()),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    tile_num_elements: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    loader_mode: LoaderMode,
}

#[cube]
impl<MP: MatmulPrecision, TO: TilingOrder> LoadingJob<MP, ContiguousTilingLayout<TO>> for Job {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, ContiguousTilingLayout<TO>>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.loader_mode == LoaderMode::Strict || this.balanced_workload) {
            load_and_store_line::<MP, TO, G>(
                this,
                unit_position,
                tensor_reader,
                stage,
                quantization,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                load_and_store_line::<MP, TO, G>(
                    this,
                    unit_position,
                    tensor_reader,
                    stage,
                    quantization,
                    config,
                );
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn load_and_store_line<MP: MatmulPrecision, TO: TilingOrder, G: GlobalConfig>(
    job: &Job,
    unit_position: u32,
    tensor_reader: &TensorReader<MP::EI>,
    stage: &mut StageMemory<MP::ES, ContiguousTilingLayout<TO>>,
    quantization: &CubeOption<Quantization<MP>>,
    #[comptime] config: G,
) {
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;

    let (tile_x, tile_y) = ContiguousTilingLayout::<TO>::to_x_y::<G::StageConfig>(
        nth_tile,
        comptime!(job.input_ident.as_ident()),
        comptime!(config.stage_config()),
    );

    let line_read = tensor_reader.load_coalesced_in_tile::<G>(
        tile_x,
        tile_y,
        pos_within_tile,
        job.input_ident,
        config,
    );

    stage.as_slice_mut(job.line_size)[unit_position / job.line_size] = match quantization {
        CubeOption::Some(quantization) => quantization.dequantize(line_read, job.input_ident),
        CubeOption::None => Line::cast_from(line_read),
    };
}
