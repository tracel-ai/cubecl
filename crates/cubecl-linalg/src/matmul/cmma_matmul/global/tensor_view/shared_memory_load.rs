use crate::matmul::cmma_matmul::stage::{
    TilingOrder, TilingOrderConfig, XMajorTiling, YMajorTiling,
};
use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
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
pub(crate) fn load_to_slice<EG: Numeric, ES: Numeric, G: GmmConfig>(
    view: &TensorView<EG>,
    slice: &mut SliceMut<'_, Line<ES>>,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) {
    // TODO allow other modes than continuous
    continuous_load_to_slice::<EG, ES, G>(view, slice, ident, config);
}

#[cube]
fn continuous_load_to_slice<EG: Numeric, ES: Numeric, G: GmmConfig>(
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

    let unit_position_base = (ContinuousSmemLoader::plane_id() * config.plane_dim()
        + ContinuousSmemLoader::plane_unit())
        * line_size;

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
            load_coalesced::<EG, G>(read_view, tile_x, tile_y, pos_within_tile, ident, config);

        slice[unit_position / line_size] = Line::cast_from(line);
    }
}

#[cube]
fn load_coalesced<EG: Numeric, G: GmmConfig>(
    this: &TensorView<EG>,
    tile_x: u32,
    tile_y: u32,
    load_id: u32,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) -> Line<EG> {
    let tensor = &this.tensor;
    let line_size = config.line_size(ident);
    let tile_size_x = config.stage_dim(ident).tile_size_x;
    let tile_size_y = config.stage_dim(ident).tile_size_y;

    let view_tile_x = tile_x * tile_size_x + this.x_offset;
    let view_tile_y = tile_y * tile_size_y + this.y_offset;

    let (load_x, load_y) = match config.layout(ident) {
        MatrixLayout::RowMajor => (load_id / tile_size_y, load_id % tile_size_y),
        MatrixLayout::ColMajor => (load_id % tile_size_x, load_id / tile_size_x),
    };

    let view_x = view_tile_x + load_x;
    let view_y = view_tile_y + load_y;

    let read_pos = (view_x * this.stride_x + view_y * this.stride_y) / line_size;

    select(
        view_x < this.shape_x && view_y < this.shape_y,
        tensor[read_pos],
        Line::empty(line_size).fill(EG::from_int(0)),
    )
}

fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
