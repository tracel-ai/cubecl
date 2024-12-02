use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{self, LoadingStrategy};
use crate::matmul::components::stage::{
    ColMajorTiling, RowMajorTiling, TilingOrder, TilingOrderConfig,
};
use crate::matmul::components::{Ident, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct CyclicLoading {}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingStrategy<EG, ES> for CyclicLoading {
    fn init_buffer<G: global::Config>(
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> SliceMut<Line<EG>> {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_buffer_elements = stage_dim.buffer_num_elements();

        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_buffer_elements / jump_length;

        #[allow(clippy::all)]
        let _ = comptime!(check_jump_divides_well(num_buffer_elements, jump_length));

        Array::vectorized(num_loads_per_unit, line_size).slice_mut(0, num_loads_per_unit)
    }

    fn fetch<G: global::Config>(
        read_view: &TensorReader<EG>,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_elements = stage_dim.total_elements();
        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_stage_elements / jump_length;

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = stage_dim.tile_num_elements();
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) = match config.tiling_order(ident) {
                TilingOrderConfig::RowMajor => RowMajorTiling::to_x_y(
                    nth_tile,
                    stage_dim.num_tiles_x_dim(),
                    stage_dim.num_tiles_y_dim(),
                ),
                TilingOrderConfig::ColMajor => ColMajorTiling::to_x_y(
                    nth_tile,
                    stage_dim.num_tiles_x_dim(),
                    stage_dim.num_tiles_y_dim(),
                ),
            };

            let line_read =
                read_view.load_coalesced::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            buffer[i] = line_read;
        }
    }

    fn store<G: global::Config>(
        buffer: &Slice<Line<EG>>,
        stage_slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_elements = stage_dim.total_elements();
        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_stage_elements / jump_length;

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;
            let line_read = buffer[i];

            match config.transpose_load(ident) {
                false => {
                    stage_slice[unit_position / line_size] = Line::cast_from(line_read);
                }
                true => {
                    let slice_line_size = config.stage_line_size(ident);

                    if comptime!(slice_line_size == 1) {
                        let tile_num_elements = stage_dim.tile_num_elements();
                        let nth_tile = unit_position / tile_num_elements;
                        let pos_within_tile = unit_position % tile_num_elements;

                        let tile_offset = nth_tile * tile_num_elements;

                        let tile_size_x = config.stage_dim(ident).tile_size_x_dim();
                        let tile_size_y = config.stage_dim(ident).tile_size_y_dim();

                        let (height, width) = match config.layout(ident) {
                            MatrixLayout::RowMajor => (tile_size_x, tile_size_y),
                            MatrixLayout::ColMajor => (tile_size_y, tile_size_x),
                        };

                        let global_strided_idx = pos_within_tile / width;
                        let global_contiguous_idx = pos_within_tile % width;

                        let slice_strided_root = global_contiguous_idx;
                        let slice_contiguous_idx = global_strided_idx;
                        let slice_stride = height;

                        #[unroll]
                        for iter in 0..config.global_line_size(ident) {
                            let slice_strided_idx = slice_strided_root + iter;
                            let elem = line_read[iter];
                            stage_slice[tile_offset
                                + slice_strided_idx * slice_stride
                                + slice_contiguous_idx] = Line::cast_from(elem);
                        }
                    } else {
                        #[allow(clippy::all)]
                        let _ = comptime!(unsupported_line_size(slice_line_size));
                    }
                }
            }
        }
    }
}

fn unsupported_line_size(line_size: u32) {
    panic!(
        "Line size for stage is not supported when transposing. Got {:?}.",
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
