use crate::matmul::components::config::InputIdent;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{self, LoadingStrategy};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of tiles from one buffer in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct BufferLoading {}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingStrategy<EG, ES> for BufferLoading {
    type LoadBuffer = Array<Line<EG>>;

    fn fetch<G: global::Config>(
        read_view: &TensorReader<EG>,
        ident: Ident,
        config: G,
    ) -> Self::LoadBuffer {
        todo!()
    }

    fn store<G: global::Config>(
        load_buffer: Self::LoadBuffer,
        stage_slice: &mut SliceMut<Line<ES>>,
        ident: Ident,
        config: G,
    ) {
        todo!()
    }
}

// #[cube]
// impl BufferLoading {
//     pub fn load_to_slice<EG: Numeric, ES: Numeric, G: global::Config>(
//         read_view: &TensorReader<EG>,
//         buffer_slice: &mut SliceMut<Line<ES>>,
//         #[comptime] num_producer_planes: u32,
//         #[comptime] producer_plane_offset: u32,
//         #[comptime] ident: Ident,
//         #[comptime] config: G,
//     ) {
//         let stage_dim = config.stage_dim(ident);
//         let line_size = config.global_line_size(ident);

//         let num_buffer_elements = stage_dim.buffer_num_elements();

//         let total_units = comptime!(num_producer_planes * config.plane_dim());
//         let jump_length = comptime!(total_units * line_size);
//         let num_loads_per_unit = num_buffer_elements / jump_length;

//         #[allow(clippy::all)]
//         let _ = comptime!(check_jump_divides_well(num_buffer_elements, jump_length));

//         let plane_id = if comptime!(producer_plane_offset > 0) {
//             UNIT_POS_Y - producer_plane_offset
//         } else {
//             UNIT_POS_Y
//         };

//         let unit_id = plane_id * config.plane_dim() + UNIT_POS_X;
//         let unit_position_base = unit_id * line_size;

//         for i in 0..num_loads_per_unit {
//             let unit_position = unit_position_base + i * jump_length;

//             let tile_num_elements = stage_dim.tile_num_elements();
//             let nth_buffer_tile = unit_position / tile_num_elements;
//             let pos_within_tile = unit_position % tile_num_elements;

//             let (tile_x, tile_y) = get_tiles_x_y(nth_buffer_tile, ident);

//             let line_read =
//                 read_view.load_coalesced::<G>(tile_x, tile_y, pos_within_tile, ident, config);

//             match config.transpose_load(ident) {
//                 false => {
//                     buffer_slice[unit_position / line_size] = Line::cast_from(line_read);
//                 }
//                 true => {
//                     #[allow(clippy::all)]
//                     let _ = comptime!(unsupported_transpose_load());
//                 }
//             }
//         }
//     }
// }

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
