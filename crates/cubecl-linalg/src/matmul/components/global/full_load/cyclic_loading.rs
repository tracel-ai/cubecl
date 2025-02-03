use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{
    ColMajorTiling, RowMajorTiling, TilingOrder, TilingOrderConfig,
};
use crate::matmul::components::{Ident, InvalidConfigError, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use pipeline::Pipeline;

use super::loader::LoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct CyclicLoading {}

pub enum LoadMode {
    Coalesced,
    Window,
}

impl LoadingValidation for CyclicLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_lines = stage_dim.total_elements() / line_size;
        let total_units = config.num_planes() * config.plane_dim();

        match config.load_mode() {
            LoadMode::Coalesced => {
                if num_stage_lines % total_units != 0 {
                    return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
                }
                if config.transpose_load(ident) && config.global_line_size(ident) != 1 {
                    return Err(Box::new(
                        "Line size for stage is not supported when transposing",
                    ));
                }
            }
            LoadMode::Window => {
                let num_slices = stage_dim.tile_size_x_dim() * stage_dim.num_tiles();
                if num_slices >= total_units && num_slices % total_units != 0 {
                    return Err(Box::new(format!("Number of units ({total_units:?}) must divide number of slices ({num_slices:?}). Would require units doing different numbers of slices")));
                }
            }
        };

        Ok(())
    }
}

#[cube]
impl LoadingStrategy for CyclicLoading {
    fn load_window<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        // pipeline: Pipeline<ES>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let total_units = config.plane_dim() * config.num_planes();
        let line_size = config.global_line_size(ident);
        let (num_slices_per_tile, slice_length_in_lines) = match config.layout(ident) {
            MatrixLayout::RowMajor => (
                stage_dim.tile_size_x_dim(),
                stage_dim.tile_size_y_dim() / line_size,
            ),
            MatrixLayout::ColMajor => (
                stage_dim.tile_size_y_dim(),
                stage_dim.tile_size_x_dim() / line_size,
            ),
        };
        stage_dim.tile_size_x_dim();
        let num_slices = comptime!(num_slices_per_tile * stage_dim.num_tiles());
        let num_slices_per_unit = num_slices.div_ceil(total_units);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        #[unroll(num_slices_per_unit==1)]
        for i in 0..num_slices_per_unit {
            let slice_index = unit_id + total_units * i;

            let nth_tile = slice_index / num_slices_per_tile;
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
            let nth_slice = slice_index % num_slices_per_tile;

            match config.transpose_load(ident) {
                false => {
                    if slice_index < num_slices {
                        let (source, this_slice_length) =
                            read_view.load_window::<G>(tile_x, tile_y, nth_slice, ident, config);

                        let slice_offset =
                            (nth_tile * num_slices_per_tile + nth_slice) * slice_length_in_lines;

                        let mut destination =
                            slice.slice_mut(slice_offset, slice_offset + this_slice_length);
                        let mut destination_pad = slice.slice_mut(
                            slice_offset + this_slice_length,
                            slice_offset + slice_length_in_lines,
                        );
                        // pipeline.memcpy_async(source.try_cast_unchecked(), destination);
                        // pipeline.memcpy_async(zero_buffer.try_cast_unchecked(), destination_offseted);
                        for i in 0..this_slice_length {
                            destination[i] = Line::cast_from(source[i]);
                        }
                        for i in 0..slice_length_in_lines - this_slice_length {
                            destination_pad[i] = Line::cast_from(0);
                        }
                    }
                }
                true => {
                    let _ = unimplemented!();
                }
            }
        }
    }
    fn load_to_slice<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);
        let num_stage_lines = stage_dim.total_elements() / line_size;
        let tile_num_lines = stage_dim.tile_num_elements() / line_size;
        let jump_length = comptime!(config.num_planes() * config.plane_dim());
        let num_loads_per_unit = comptime!(num_stage_lines / jump_length);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_id + i * jump_length;

            let nth_tile = unit_position / tile_num_lines;
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

            let pos_within_tile = (unit_position % tile_num_lines) * line_size;

            let line_read =
                read_view.load_coalesced::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            match config.transpose_load(ident) {
                false => {
                    slice[unit_position] = Line::cast_from(line_read);
                }
                true => {
                    let tile_offset = nth_tile * tile_num_lines * line_size;

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
