use crate::matmul::cmma_matmul::global::load_coalesced;
use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matmul_stage::{TilingOrder, XMajorTiling};
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::TensorView;

#[derive(CubeType, Clone, Copy)]
pub struct ContinuousSmemLoader {}

#[cube]
impl PlaneMapper for ContinuousSmemLoader {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
pub(crate) fn load_shared_memory<EG: Numeric, ES: Numeric, G: GmmConfig>(
    read_view: &TensorView<EG>,
    smem: &mut SharedMemory<Line<ES>>,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) {
    let stage_dim = config.stage_dim(ident);
    let line_size = config.line_size(ident);
    let num_stage_elements = stage_dim.num_elements();

    let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
    let _ = comptime!(check_jump_divides_well(num_stage_elements, jump_length));

    let unit_position_base = (ContinuousSmemLoader::plane_id() * config.plane_dim()
        + ContinuousSmemLoader::plane_unit())
        * line_size;

    for i in 0..num_stage_elements / jump_length {
        let unit_position = unit_position_base + i * jump_length;

        let tile_num_elements = stage_dim.tile_num_elements();
        let nth_tile = unit_position / tile_num_elements;
        let pos_within_tile = unit_position % tile_num_elements;

        // TODO X or Y choose with comptime config
        let (tile_x, tile_y) =
            XMajorTiling::to_x_y(nth_tile, stage_dim.num_tiles_x, stage_dim.num_tiles_y);

        let line =
            load_coalesced::<EG, G>(read_view, tile_x, tile_y, pos_within_tile, ident, config);

        smem[unit_position / line_size] = Line::cast_from(line);
    }
}

fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
