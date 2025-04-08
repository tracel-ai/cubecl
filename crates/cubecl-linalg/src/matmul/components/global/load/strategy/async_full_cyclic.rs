use std::marker::PhantomData;

use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation, Quantization,
        load::AsyncFullLoadingStrategy, tensor_view::TensorReader,
    },
    stage::{ContiguousTilingLayout, Stage, TilingOrder},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};
use cubecl_std::CubeOption;

use super::LoadingJob;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct AsyncFullCyclicLoading<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<(MP, CM, T)>,
}

impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, T: TilingOrder> LoadingValidation
    for AsyncFullCyclicLoading<MP, CM, T>
{
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
impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, T: TilingOrder>
    AsyncFullLoadingStrategy<MP, CM> for AsyncFullCyclicLoading<MP, CM, T>
{
    type TilingLayout = ContiguousTilingLayout<T>;
    type Job = AsyncFullCyclicJob<MP, CM, T>;

    fn load_full<G: GlobalConfig>(
        read_view: TensorReader<MP::EI>,
        mut stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        _quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        let stage_dim = config.tiling_dimensions(input_ident);
        let total_units = config.plane_dim() * config.num_planes();
        let line_size = config.global_line_size(input_ident);

        let (num_slices_per_tile, slice_length_in_lines) = match config.matrix_layout(input_ident) {
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
                input_ident.as_ident(),
                config.to_smm_config(),
            );
            let nth_slice = slice_index % num_slices_per_tile;

            // TODO make branching comptime conditional
            if slice_index < num_slices {
                let window = read_view.load_window_in_tile::<G>(
                    (tile_x, tile_y),
                    nth_slice,
                    input_ident,
                    config,
                );

                // Where this unit writes source in the stage
                let slice_destination_offset =
                    (nth_tile * num_slices_per_tile + nth_slice) * slice_length_in_lines;

                // Make destination start at offset
                let mut destination = stage.as_slice_mut().slice_mut(
                    slice_destination_offset,
                    slice_destination_offset + slice_length_in_lines,
                );

                CM::memcpy_async(
                    &mechanism,
                    &window.slice.try_cast_unchecked(),
                    &mut destination,
                );
            }
        }
    }

    fn job<G: GlobalConfig>(
        read_view: TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        _quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> AsyncFullCyclicJob<MP, CM, T> {
        todo!()
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullCyclicJob<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, T: TilingOrder> {
    read_view: TensorReader<MP::EI>,
    stage: Stage<MP::ES, ContiguousTilingLayout<T>>,
    mechanism: CM,
    _quantization: CubeOption<Quantization<MP>>,
}

#[cube]
impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, T: TilingOrder> LoadingJob<MP>
    for AsyncFullCyclicJob<MP, CM, T>
{
    fn len(_this: &Self) -> u32 {
        todo!()
    }

    fn execute_task<G: GlobalConfig>(this: &mut Self, task_id: u32, #[comptime] config: G) {
        // TODO
    }
}
