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
pub struct CyclicLoading {}

#[cube]
impl PlaneMapper for CyclicLoading {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl CyclicLoading {
    pub fn load_to_slice<EG: Numeric, ES: Numeric, G: Config>(
        read_view: &TensorView<EG>,
        slice: &mut SliceMut<'_, Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_elements = stage_dim.num_elements();
        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_stage_elements / jump_length;

        #[allow(clippy::all)]
        let _ = comptime!(check_jump_divides_well(num_stage_elements, jump_length));

        let unit_id = Self::plane_id() * config.plane_dim() + Self::plane_unit();
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
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

            let line_read =
                read_view.load_coalesced::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            match config.transpose_load(ident) {
                false => {
                    slice[unit_position / line_size] = Line::cast_from(line_read);
                }
                true => {
                    // TODO get comptime
                    let slice_line_size = 1;

                    if comptime!(slice_line_size == 1) {
                        let tile_offset = nth_tile * tile_num_elements;

                        // let (load_x, load_y) = match config.layout(ident) {
                        //     MatrixLayout::RowMajor => (unit_id / tile_size_y, unit_id % tile_size_y),
                        //     MatrixLayout::ColMajor => (unit_id % tile_size_x, unit_id / tile_size_x),
                        // };
                        let tile_size_x = config.stage_dim(ident).tile_size_x;
                        let tile_size_y = config.stage_dim(ident).tile_size_y;

                        let row_in_global = pos_within_tile / tile_size_y;
                        let root_col_in_global = pos_within_tile % tile_size_y;

                        let root_row_in_slice = root_col_in_global;
                        let col_in_slice = row_in_global;

                        let stride_row = tile_size_x;
                        let stride_col = 1;

                        #[unroll]
                        for row_iter in 0..config.global_line_size(ident) {
                            let row_in_slice = root_row_in_slice + row_iter;
                            let elem = line_read[row_iter];
                            slice[tile_offset
                                + row_in_slice * stride_row
                                + col_in_slice * stride_col] = Line::cast_from(elem);
                        }
                    } else {
                        let _ = comptime!(unsupported_line_size(slice_line_size));
                    }
                }
            }
        }
    }
}

fn unsupported_line_size(line_size: u32) {
    panic!(
        "If load must transpose, then lined stage is not supported. Got {:?}",
        line_size
    )
}

fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
