use crate::matmul::components::config::PlaneMapper;
use crate::matmul::components::global::Config;
use crate::matmul::components::stage::{
    TilingOrder, TilingOrderConfig, XMajorTiling, YMajorTiling,
};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::TensorView;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct ContinuousLoading {}

#[cube]
impl PlaneMapper for ContinuousLoading {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl ContinuousLoading {
    pub fn load_to_slice<EG: Numeric, ES: Numeric, G: Config>(
        read_view: &TensorView<EG>,
        slice: &mut SliceMut<'_, Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.line_size(ident);
        let num_stage_elements = stage_dim.num_elements();

        let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
        let _ = comptime!(check_jump_divides_well(num_stage_elements, jump_length));

        let unit_position_base =
            (Self::plane_id() * config.plane_dim() + Self::plane_unit()) * line_size;

        for i in 0..num_stage_elements / jump_length {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = stage_dim.tile_num_elements();
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) = match config.tiling_order() {
                TilingOrderConfig::XMajor => {
                    XMajorTiling::to_x_y(nth_tile, stage_dim.num_tiles_x, stage_dim.num_tiles_y)
                }
                TilingOrderConfig::YMajor => {
                    YMajorTiling::to_x_y(nth_tile, stage_dim.num_tiles_x, stage_dim.num_tiles_y)
                }
            };

            let line =
                read_view.load_coalesced::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            slice[unit_position / line_size] = Line::cast_from(line);
        }
    }
}

fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
