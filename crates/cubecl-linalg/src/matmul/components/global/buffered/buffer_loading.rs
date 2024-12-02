use std::marker::PhantomData;

use crate::matmul::components::config::InputIdent;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{self, LoadingStrategy};
use crate::matmul::components::stage::{Stage, TilingOrderConfig};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of tiles from one buffer in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct BufferLoading<EG: Numeric, ES: Numeric> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingStrategy<EG, ES> for BufferLoading<EG, ES> {
    type LoadBuffer = Array<Line<EG>>;

    fn fetch<G: global::Config>(
        read_view: &TensorReader<EG>,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Self::LoadBuffer {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_buffer_elements = stage_dim.buffer_num_elements();

        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_buffer_elements / jump_length;

        #[allow(clippy::all)]
        let _ = comptime!(check_jump_divides_well(num_buffer_elements, jump_length));

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        let mut load_buffer = Array::vectorized(num_loads_per_unit, line_size);

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = stage_dim.tile_num_elements();
            let nth_buffer_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) = get_tiles_x_y(nth_buffer_tile, ident);

            let line_read =
                read_view.load_coalesced::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            load_buffer[i] = line_read;
        }

        load_buffer
    }

    fn store<G: global::Config>(
        buffer: &Slice<Line<EG>>,
        stage_slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_buffer_elements = stage_dim.buffer_num_elements();

        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_buffer_elements / jump_length;

        #[allow(clippy::all)]
        let _ = comptime!(check_jump_divides_well(num_buffer_elements, jump_length));

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            match config.transpose_load(ident) {
                false => {
                    stage_slice[unit_position / line_size] = Line::cast_from(load_buffer[i]);
                }
                true => {
                    #[allow(clippy::all)]
                    let _ = comptime!(unsupported_transpose_load());
                }
            }
        }
    }
}

#[cube]
fn get_tiles_x_y(nth_buffer_tile: u32, #[comptime] ident: Ident) -> (u32, u32) {
    match comptime!(ident.as_input()) {
        InputIdent::Lhs => {
            // Assuming ColMajor tiling order
            (nth_buffer_tile, 0)
        }
        InputIdent::Rhs => {
            // Assuming RowMajor tiling order
            (0, nth_buffer_tile)
        }
    }
}

#[cube]
pub(crate) fn buffer_slice<EG: Numeric, ES: Numeric, G: global::Config>(
    buffer_iter: u32,
    stage: &mut Stage<ES>,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) -> SliceMut<Line<ES>> {
    let buffer_num_elements = config.stage_dim(ident).buffer_num_elements();
    let line_size = config.stage_line_size(ident);
    let buffer_num_lines = buffer_num_elements / line_size;

    #[allow(clippy::all)]
    let _ = comptime!(check_buffers_contiguous(ident, config));

    let start = buffer_iter * buffer_num_lines;
    let end = start + buffer_num_lines;
    stage.as_slice_mut().slice_mut(start, end)
}

fn check_buffers_contiguous<G: global::Config>(ident: Ident, config: G) {
    match ident.as_input() {
        InputIdent::Lhs => {
            if let TilingOrderConfig::RowMajor = config.tiling_order(ident) {
                panic!("Lhs must have ColMajor tiling order in pipelined setting")
            }
        }
        InputIdent::Rhs => {
            if let TilingOrderConfig::ColMajor = config.tiling_order(ident) {
                panic!("Rhs must have RowMajor tiling order in pipelined setting")
            }
        }
    }
}

fn unsupported_transpose_load() {
    panic!("Transpose load not yet supported in buffered setup")
}

fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
