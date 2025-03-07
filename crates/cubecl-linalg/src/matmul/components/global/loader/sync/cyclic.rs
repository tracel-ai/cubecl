use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{ContiguousTilingLayout, Stage, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SyncLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct CyclicCoalescedLoading<T: TilingOrder> {
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for CyclicCoalescedLoading<T> {
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
impl<T: TilingOrder> SyncLoadingStrategy for CyclicCoalescedLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;

    fn load_full<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);
        let num_stage_elements = tiling.total_size();
        let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
        let num_loads_per_unit = comptime!(num_stage_elements / jump_length);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = tiling.tile_size();
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) = ContiguousTilingLayout::<T>::to_x_y_from_nth::<G::SmmConfig>(
                nth_tile,
                ident,
                config.to_smm_config(),
            );

            let line_read = read_view.load_coalesced_in_tile::<G>(
                tile_x,
                tile_y,
                pos_within_tile,
                ident,
                config,
            );

            stage.as_slice_mut()[unit_position / line_size] = Line::cast_from(line_read);
        }
    }

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        buffer_index: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let tiling_dimensions = config.tiling_dimensions(ident);
        let num_lines_per_tile = tiling_dimensions.tile_size() / config.stage_line_size(ident);
        let tile_count_row = tiling_dimensions.tile_count_row();
        let tile_count_col = tiling_dimensions.tile_count_col();
        let total_units = config.plane_dim() * config.num_planes();
        let unit_base = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        let num_tiles_in_buffer = match ident.as_input() {
            InputIdent::Lhs => tile_count_row,
            InputIdent::Rhs => tile_count_col,
        };

        let total_num_lines = num_tiles_in_buffer * num_lines_per_tile;
        let num_lines_per_unit = (total_num_lines + total_units - 1) / total_units;

        for i in 0..num_lines_per_unit {
            let nth_line = unit_base + i * total_units;
            if nth_line < total_num_lines {
                let nth_tile_in_buffer = nth_line / num_lines_per_tile;
                let pos_within_tile = nth_line % num_lines_per_tile;

                let (tile_x, tile_y) = match ident.as_input() {
                    InputIdent::Lhs => (nth_tile_in_buffer, buffer_index),
                    InputIdent::Rhs => (buffer_index, nth_tile_in_buffer),
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

                tile_slice[pos_within_tile] = Line::cast_from(line_read);
            }
        }
    }
}
