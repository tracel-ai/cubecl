use std::marker::PhantomData;

use crate::matmul::components::global::load::SyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, StageMemory, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoaderMode, LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for LoadingStrategy<TO> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        if let LoaderMode::Strict = config.loader_mode() {
            let line_size = config.global_line_size(ident);
            let num_lines_per_tile = config.tiling_scheme().elements_in_tile(ident) / line_size;
            let num_tiles_in_buffer = config.tiling_scheme().tiles_in_stage(ident);
            let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;

            let total_units = config.plane_dim() * config.num_planes();
            let jump_length = total_units * line_size;
            let num_tasks_per_unit = total_num_lines.div_ceil(total_units);

            let max_id = total_units - 1;
            let max_task_id = num_tasks_per_unit - 1;
            let max_position_base = max_id * line_size;
            let max_position = max_position_base + max_task_id * jump_length;
            let num_stage_elements = config.tiling_scheme().elements_in_stage(ident);

            if max_position > num_stage_elements {
                return Err(Box::new(
                    "Too many data will be loaded, resulting in out-of-bounds",
                ));
            }
        }

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
        let line_size = config.global_line_size(input_ident);
        let num_stage_elements = config.tiling_scheme().elements_in_stage(input_ident);
        let tile_size = config.tiling_scheme().elements_in_tile(input_ident);
        let tile_count_row = config.tiling_scheme().tiles_in_stage_row(input_ident);
        let tile_count_col = config.tiling_scheme().tiles_in_stage_col(input_ident);

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
    let (line_size, tile_size, tile_count_row, tile_count_col) = comptime! {
        (
            config.global_line_size(job.input_ident),
            config.tiling_scheme().elements_in_tile(job.input_ident),
            config.tiling_scheme().tiles_in_stage_row(job.input_ident),
            config.tiling_scheme().tiles_in_stage_col(job.input_ident),
        )
    };

    let tile_index = unit_position / tile_size;
    let pos_within_tile = unit_position % tile_size;

    let (total_tile_count_row, total_tile_count_col) = match comptime!(job.input_ident) {
        InputIdent::Lhs => (
            comptime!(tile_count_row),
            comptime!(tile_count_col * config.num_stages(InputIdent::Lhs)),
        ),
        InputIdent::Rhs => (
            comptime!(tile_count_row * config.num_stages(InputIdent::Rhs)),
            comptime!(tile_count_col),
        ),
    };

    let (tile_x_within_buffer, tile_y_within_buffer) = TO::to_row_col::<G::SmmConfig>(
        tile_index,
        tile_count_row,
        tile_count_col,
        comptime!(job.input_ident.as_ident()),
        comptime!(config.to_smm_config()),
    );

    let (tile_x, tile_y) = match comptime!(job.input_ident) {
        InputIdent::Lhs => (
            tile_x_within_buffer,
            job.buffer_index * tile_count_col + tile_y_within_buffer,
        ),
        InputIdent::Rhs => (
            job.buffer_index * tile_count_row + tile_x_within_buffer,
            tile_y_within_buffer,
        ),
    };

    let line_read = tensor_reader.load_coalesced_in_tile::<G>(
        tile_x,
        tile_y,
        pos_within_tile,
        job.input_ident,
        config,
    );

    let nth_tile_in_stage = TO::to_nth_tile::<G::SmmConfig>(
        tile_x,
        tile_y,
        total_tile_count_row,
        total_tile_count_col,
        comptime!(job.input_ident.as_ident()),
        config.to_smm_config(),
    );

    let tile_start = nth_tile_in_stage * job.num_lines_per_tile;
    let tile_end = tile_start + job.num_lines_per_tile;
    let mut tile_slice = stage
        .as_slice_mut(line_size)
        .slice_mut(tile_start, tile_end);

    tile_slice[pos_within_tile / line_size] = match quantization {
        CubeOption::Some(quantization) => quantization.dequantize(line_read, job.input_ident),
        CubeOption::None => Line::cast_from(line_read),
    }
}
