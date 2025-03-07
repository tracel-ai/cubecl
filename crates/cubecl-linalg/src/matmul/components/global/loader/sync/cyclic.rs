use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{ContiguousTilingLayout, TilingOrder};
use crate::matmul::components::{Ident, InvalidConfigError, MatrixLayout};
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
        if config.transpose_load(ident) && config.global_line_size(ident) != 1 {
            return Err(Box::new(
                "Line size for stage is not supported when transposing",
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
        slice: &mut SliceMut<Line<ES>>,
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

            let (tile_x, tile_y) = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
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

            match config.transpose_load(ident) {
                false => {
                    slice[unit_position / line_size] = Line::cast_from(line_read);
                }
                true => {
                    let tile_offset = nth_tile * tile_num_elements;

                    let tile_shape_x = config.tiling_dimensions(ident).tile_shape_row();
                    let tile_shape_y = config.tiling_dimensions(ident).tile_shape_col();

                    let (height, width) = match config.matrix_layout(ident) {
                        MatrixLayout::RowMajor => (tile_shape_x, tile_shape_y),
                        MatrixLayout::ColMajor => (tile_shape_y, tile_shape_x),
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

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage_slice: &mut SliceMut<Line<ES>>,
        buffer_index: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        // TODO
    }
}
