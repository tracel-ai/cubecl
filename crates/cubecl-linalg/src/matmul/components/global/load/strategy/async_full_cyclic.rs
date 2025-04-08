use std::marker::PhantomData;

use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation, Quantization,
        load::{AsyncFullLoadingStrategy, default_async_full_load},
        tensor_view::TensorReader,
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
pub struct AsyncFullCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for AsyncFullCyclicLoading<T> {
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
impl<T: TilingOrder> AsyncFullLoadingStrategy for AsyncFullCyclicLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;
    type Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> = AsyncFullCyclicJob<MP, CM, T>;

    fn load_full<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        default_async_full_load::<Self, MP, G, CM>(
            read_view,
            stage,
            mechanism,
            quantization,
            input_ident,
            config,
        )
    }

    fn job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        _quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> AsyncFullCyclicJob<MP, CM, T> {
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
        let num_tasks = num_slices.div_ceil(total_units);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        AsyncFullCyclicJob::<MP, CM, T> {
            unit_id,
            stage,
            mechanism,
            num_tasks,
            total_units,
            num_slices,
            input_ident,
            num_slices_per_tile,
            slice_length_in_lines,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullCyclicJob<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, T: TilingOrder> {
    unit_id: u32,

    stage: Stage<MP::ES, ContiguousTilingLayout<T>>,
    mechanism: CM,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    total_units: u32,
    #[cube(comptime)]
    num_slices: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    num_slices_per_tile: u32,
    #[cube(comptime)]
    slice_length_in_lines: u32,
}

#[cube]
impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, T: TilingOrder> LoadingJob<MP>
    for AsyncFullCyclicJob<MP, CM, T>
{
    fn len(this: &Self) -> u32 {
        this.num_tasks.runtime()
    }

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        read_view: &TensorReader<MP::EI>,
        #[comptime] config: G,
    ) {
        let slice_index = this.unit_id + this.total_units * task_id;

        let nth_tile = slice_index / this.num_slices_per_tile;
        let (tile_x, tile_y) = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
            nth_tile,
            comptime!(this.input_ident.as_ident()),
            config.to_smm_config(),
        );
        let nth_slice = slice_index % this.num_slices_per_tile;

        // TODO make branching comptime conditional
        if slice_index < this.num_slices {
            let window = read_view.load_window_in_tile::<G>(
                (tile_x, tile_y),
                nth_slice,
                this.input_ident,
                config,
            );

            // Where this unit writes source in the stage
            let slice_destination_offset =
                (nth_tile * this.num_slices_per_tile + nth_slice) * this.slice_length_in_lines;

            // Make destination start at offset
            let mut destination = this.stage.as_slice_mut().slice_mut(
                slice_destination_offset,
                slice_destination_offset + this.slice_length_in_lines,
            );

            CM::memcpy_async(
                &this.mechanism,
                &window.slice.try_cast_unchecked(),
                &mut destination,
            );
        }
    }
}
