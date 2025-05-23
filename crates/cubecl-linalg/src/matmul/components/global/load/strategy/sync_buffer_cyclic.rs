use std::marker::PhantomData;

use crate::matmul::components::global::load::SyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, StageMemory, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for LoadingStrategy<TO> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling_dimensions = config.tiling_dimensions(ident);
        let num_stage_elements = tiling_dimensions.total_size();
        let line_size = config.global_line_size(ident);
        let tile_size = tiling_dimensions.tile_size();
        let tile_count_row = tiling_dimensions.tile_count_row();
        let tile_count_col = tiling_dimensions.tile_count_col();
        let num_lines_per_tile = tile_size / line_size;

        let total_units = config.plane_dim() * config.num_planes();
        let jump_length = total_units * line_size;
        let num_tiles_in_buffer = tile_count_row * tile_count_col;

        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let max_id = total_units - 1;
        let num_tasks_per_unit = total_num_lines.div_ceil(total_units);
        let max_task = num_tasks_per_unit - 1;
        let max_position_base = max_id * line_size;
        let max_position = max_position_base + max_task * jump_length;

        comptime! {
        println!("---");
        println!("ident: {:?}", ident);
        println!("max_position: {:?}", max_position);
        println!("num_stage_elements: {:?}", num_stage_elements);
        }

        if max_position > num_stage_elements {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out-of-bounds",
            ));
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
        let tiling_dimensions = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);
        let tile_size = tiling_dimensions.tile_size();
        let tile_count_row = tiling_dimensions.tile_count_row();
        let tile_count_col = tiling_dimensions.tile_count_col();

        let num_lines_per_tile = tile_size / line_size;
        let total_units = config.plane_dim() * config.num_planes();
        let jump_length = total_units * line_size;

        let num_tiles_in_buffer = tile_count_row * tile_count_col;
        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let num_tasks_per_unit = (total_num_lines + total_units - 1) / total_units;

        comptime! {
        println!("---");
        println!("ident: {:?}", input_ident);
        println!("total_units: {:?}", total_units);
        println!("jump_length: {:?}", jump_length);
        println!("num_tiles_in_buffer: {:?}", num_tiles_in_buffer);
        println!("total_num_lines: {:?}", total_num_lines);
        println!("num_tasks_per_unit: {:?}", num_tasks_per_unit);
        }

        //max=511
        // let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_id = UNIT_POS;
        if unit_id == 65535 {
            terminate!()
        }
        let unit_position_base = unit_id * line_size;

        Job {
            unit_position_base,
            num_tasks_per_unit,
            buffer_index,
            jump_length,
            num_lines_per_tile,
            input_ident,
            num_stage_elements: tiling_dimensions.total_size(),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    pub unit_position_base: u32,

    #[cube(comptime)]
    pub num_tasks_per_unit: u32,
    #[cube(comptime)]
    pub buffer_index: u32,
    #[cube(comptime)]
    pub jump_length: u32,
    #[cube(comptime)]
    pub num_lines_per_tile: u32,
    #[cube(comptime)]
    pub input_ident: InputIdent,
    // tmp
    #[cube(comptime)]
    pub num_stage_elements: u32,
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

        match comptime!(this.input_ident) {
            InputIdent::Lhs => {
                load_and_store_line::<MP, TO, G>(
                    this,
                    unit_position,
                    tensor_reader,
                    stage,
                    quantization,
                    config,
                );
            }
            InputIdent::Rhs => {
                // 128 * 2048 = 262144
                // 262140 pass
                // 262144 fail
                // if unit_position < 262148 {
                if this.unit_position_base < 262140 {
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
        let tiling_dimensions = config.tiling_dimensions(job.input_ident);
        (
            config.global_line_size(job.input_ident),
            tiling_dimensions.tile_size(),
            tiling_dimensions.tile_count_row(),
            tiling_dimensions.tile_count_col()
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
