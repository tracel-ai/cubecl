use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{ContiguousTilingLayout, Stage, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SyncBufferLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct CyclicCoalescedBufferLoading<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for CyclicCoalescedBufferLoading<T> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling_dimensions = config.tiling_dimensions(ident);
        let line_size = config.stage_line_size(ident);
        let tile_size = tiling_dimensions.tile_size();
        let tile_count_row = tiling_dimensions.tile_count_row();
        let tile_count_col = tiling_dimensions.tile_count_col();
        let num_lines_per_tile = tile_size / line_size;

        let total_units = config.plane_dim() * config.num_planes();
        let jump_length = total_units * line_size;
        let num_tiles_in_buffer = comptime! {match ident.as_input() {
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
impl<T: TilingOrder> SyncBufferLoadingStrategy for CyclicCoalescedBufferLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        buffer_index: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let tiling_dimensions = config.tiling_dimensions(ident);
        let line_size = config.stage_line_size(ident);
        let tile_size = tiling_dimensions.tile_size();
        let tile_count_row = tiling_dimensions.tile_count_row();
        let tile_count_col = tiling_dimensions.tile_count_col();

        let num_lines_per_tile = tile_size / line_size;
        let total_units = config.plane_dim() * config.num_planes();
        let jump_length = total_units * line_size;

        let num_tiles_in_buffer = comptime! {match ident.as_input() {
            InputIdent::Lhs => tile_count_row,
            InputIdent::Rhs => tile_count_col,
        }};
        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let num_lines_per_unit = (total_num_lines + total_units - 1) / total_units;

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_lines_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            // We assume unit_position < total_num_lines * line_size;
            // This is caught by the loading validation

            let unit_pos_in_buffer = unit_position / tile_size;
            let pos_within_tile = unit_position % tile_size;

            let (tile_x, tile_y) = match ident.as_input() {
                InputIdent::Lhs => (unit_pos_in_buffer, buffer_index),
                InputIdent::Rhs => (buffer_index, unit_pos_in_buffer),
            };

            let nth_tile = T::to_nth_tile(tile_x, tile_y, tile_count_row, tile_count_col);

            let line_read = read_view.load_coalesced_in_tile::<G>(
                tile_x,
                tile_y,
                pos_within_tile,
                ident,
                config,
            );

            let tile_start = nth_tile * num_lines_per_tile;
            let tile_end = tile_start + num_lines_per_tile;
            let mut tile_slice = stage.as_slice_mut().slice_mut(tile_start, tile_end);

            tile_slice[pos_within_tile / line_size] = Line::cast_from(line_read);
        }
    }
}
