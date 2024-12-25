use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::stage::{
    ColMajorTiling, RowMajorTiling, TilingOrder, TilingOrderConfig,
};
use crate::matmul::components::{Ident, InvalidConfigError, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::loader::LoadingStrategy;
use super::LoadingValidation;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct CyclicLoading {}

impl LoadingValidation for CyclicLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_elements = stage_dim.total_elements();
        let total_units = config.num_planes() * config.plane_dim();
        let jump_length = total_units * line_size;

        if num_stage_elements % jump_length != 0 {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that jump_length divides num_stage_elements.",
            ));
        }

        if config.transpose_load(ident) && config.global_line_size(ident) != 1 {
            return Err(Box::new(
                "Line size for stage is not supported when transposing",
            ));
        }

        Ok(())
    }
}

#[cube]
impl LoadingStrategy for CyclicLoading {
    fn load_to_slice<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
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

            match config.transpose_load(ident) {
                false => {
                    slice[unit_position / line_size] = Line::cast_from(line_read);
                }
                true => {
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
                        slice[tile_offset
                            + slice_strided_idx * slice_stride
                            + slice_contiguous_idx] = Line::cast_from(elem);
                    }
                }
            }
        }
    }
}
