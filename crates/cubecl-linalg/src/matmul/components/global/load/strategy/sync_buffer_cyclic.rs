use std::marker::PhantomData;

use crate::matmul::components::global::load::SyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, Stage, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::LoadingJob;

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
        let line_size = config.global_line_size(ident);
        let tile_size = tiling_dimensions.tile_size();
        let tile_count_row = tiling_dimensions.tile_count_row();
        let tile_count_col = tiling_dimensions.tile_count_col();
        let num_lines_per_tile = tile_size / line_size;

        let total_units = config.plane_dim() * config.num_planes();
        let jump_length = total_units * line_size;
        let num_tiles_in_buffer = comptime! {match ident.as_input_ident() {
            InputIdent::Lhs => tile_count_row,
            InputIdent::Rhs => tile_count_col,
        }};
        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let num_lines_per_unit = (total_num_lines + total_units - 1) / total_units;

        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let out_of_bounds_pos = total_num_lines * line_size;

        let max_id = total_units - 1;
        let max_iter = num_lines_per_unit - 1;
        let max_position_base = max_id * line_size;
        let max_position = max_position_base + max_iter * jump_length;

        if max_position > out_of_bounds_pos {
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

        let num_tiles_in_buffer = comptime! {match input_ident {
            InputIdent::Lhs => tile_count_row,
            InputIdent::Rhs => tile_count_col,
        }};
        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let num_tasks = (total_num_lines + total_units - 1) / total_units;

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        Job {
            unit_position_base,
            num_tasks,
            buffer_index,
            jump_length,
            num_lines_per_tile,
            input_ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    buffer_index: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    num_lines_per_tile: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision, TO: TilingOrder> LoadingJob<MP, ContiguousTilingLayout<TO>> for Job {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, ContiguousTilingLayout<TO>>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let (line_size, tile_size, tile_count_row, tile_count_col) = comptime! {
            let tiling_dimensions = config.tiling_dimensions(this.input_ident);
            (
                config.global_line_size(this.input_ident),
                tiling_dimensions.tile_size(),
                tiling_dimensions.tile_count_row(),
                tiling_dimensions.tile_count_col()
            )
        };

        let unit_position = this.unit_position_base + task_id * this.jump_length;

        // We assume unit_position < total_num_lines * line_size;
        // This is caught by the loading validation

        let unit_pos_in_buffer = unit_position / tile_size;
        let pos_within_tile = unit_position % tile_size;

        let (tile_x, tile_y) = match comptime!(this.input_ident) {
            InputIdent::Lhs => (unit_pos_in_buffer, this.buffer_index.runtime()),
            InputIdent::Rhs => (this.buffer_index.runtime(), unit_pos_in_buffer),
        };

        let nth_tile = TO::to_nth_tile::<G::SmmConfig>(
            tile_x,
            tile_y,
            tile_count_row,
            tile_count_col,
            comptime!(this.input_ident.as_ident()),
            config.to_smm_config(),
        );

        let x: u32 = todo!();
        let line_read = tensor_reader.load_coalesced_in_tile::<G>(
            tile_x,
            tile_y,
            // pos_within_tile,
            x,
            x,
            this.input_ident,
            config,
        );

        let tile_start = nth_tile * this.num_lines_per_tile;
        let tile_end = tile_start + this.num_lines_per_tile;
        let mut tile_slice = stage
            .as_slice_mut(line_size)
            .slice_mut(tile_start, tile_end);

        tile_slice[pos_within_tile / line_size] = match quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read, this.input_ident),
            CubeOption::None => Line::cast_from(line_read),
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
