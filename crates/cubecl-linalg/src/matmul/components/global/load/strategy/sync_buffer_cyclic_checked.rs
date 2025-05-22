use std::marker::PhantomData;

use crate::matmul::components::global::load::{SyncBufferLoadingStrategy, sync_buffer_cyclic};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, StageMemory, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for LoadingStrategy<TO> {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> SyncBufferLoadingStrategy for LoadingStrategy<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] buffer_index: u32,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job {
        let tiling_dimensions = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);
        let num_stage_elements = tiling_dimensions.total_size();
        let tile_size = tiling_dimensions.tile_size();
        let tile_count_row = tiling_dimensions.tile_count_row();
        let tile_count_col = tiling_dimensions.tile_count_col();

        let num_lines_per_tile = tile_size / line_size;
        let total_units = config.plane_dim() * config.num_planes();
        let jump_length = total_units * line_size;

        let num_tiles_in_buffer = tile_count_row * tile_count_col;
        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let num_tasks_per_unit = total_num_lines.div_ceil(total_units);
        let balanced_workload = num_tasks_per_unit % total_units == 0;

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        Job {
            unit_position_base,
            num_tasks_per_unit,
            buffer_index,
            jump_length,
            num_lines_per_tile,
            input_ident,
            balanced_workload,
            num_stage_elements,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    buffer_index: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    num_lines_per_tile: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
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

        let buffer_cyclic_job = sync_buffer_cyclic::Job {
            unit_position_base: this.unit_position_base,
            num_tasks_per_unit: comptime!(this.num_tasks_per_unit),
            buffer_index: comptime!(this.buffer_index),
            jump_length: comptime!(this.jump_length),
            num_lines_per_tile: comptime!(this.num_lines_per_tile),
            input_ident: comptime!(this.input_ident),
        };

        #[allow(clippy::collapsible_else_if)]
        if this.balanced_workload {
            sync_buffer_cyclic::load_and_store_line::<MP, TO, G>(
                &buffer_cyclic_job,
                unit_position,
                tensor_reader,
                stage,
                quantization,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                sync_buffer_cyclic::load_and_store_line::<MP, TO, G>(
                    &buffer_cyclic_job,
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
