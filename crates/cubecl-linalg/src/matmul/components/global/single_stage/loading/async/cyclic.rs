use std::marker::PhantomData;

use crate::matmul::components::{
    Ident, InvalidConfigError, MatrixLayout,
    global::{CopyMechanism, GlobalConfig, LoadingValidation, tensor_view::TensorReader},
    stage::{ContiguousTilingLayout, Stage, TilingOrder},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::AsyncFullLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct CyclicWindowLoading<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for CyclicWindowLoading<T> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.tiling_dimensions(ident);
        let total_units = config.num_planes() * config.plane_dim();

        let num_slices = tiling.tile_shape_row() * tiling.tile_count();
        if num_slices >= total_units && num_slices % total_units != 0 {
            return Err(Box::new(format!(
                "Number of units ({total_units:?}) must divide number of slices ({num_slices:?}). Would require units doing different numbers of slices"
            )));
        }

        Ok(())
    }
}

#[cube]
impl<T: TilingOrder> AsyncFullLoadingStrategy for CyclicWindowLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;

    fn load_full<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.tiling_dimensions(ident);
        let total_units = config.plane_dim() * config.num_planes();
        let line_size = config.global_line_size(ident);

        let (num_slices_per_tile, slice_length_in_lines) = match config.matrix_layout(ident) {
            MatrixLayout::RowMajor => (
                stage_dim.tile_shape_row(),
                stage_dim.tile_shape_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                stage_dim.tile_shape_col(),
                stage_dim.tile_shape_row() / line_size,
            ),
        };

        let num_slices = comptime!(num_slices_per_tile * stage_dim.tile_count());
        let num_slices_per_unit = num_slices.div_ceil(total_units);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        #[unroll(num_slices_per_unit==1)]
        for i in 0..num_slices_per_unit {
            let slice_index = unit_id + total_units * i;

            let nth_tile = slice_index / num_slices_per_tile;
            let (tile_x, tile_y) = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
                nth_tile,
                ident,
                config.to_smm_config(),
            );
            let nth_slice = slice_index % num_slices_per_tile;

            // TODO make branching comptime conditional
            if slice_index < num_slices {
                let window =
                    read_view.load_window_in_tile::<G>((tile_x, tile_y), nth_slice, ident, config);

                // Where this unit writes source in the stage
                let slice_destination_offset =
                    (nth_tile * num_slices_per_tile + nth_slice) * slice_length_in_lines;

                // Make destination start at offset
                let mut destination = stage.as_slice_mut().slice_mut(
                    slice_destination_offset,
                    slice_destination_offset + slice_length_in_lines,
                );

                CM::memcpy_async(
                    mechanism,
                    &window.slice.try_cast_unchecked(),
                    &mut destination,
                );
            }
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}
