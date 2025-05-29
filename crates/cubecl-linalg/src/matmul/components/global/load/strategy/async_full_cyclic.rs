use std::marker::PhantomData;

use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, load::AsyncFullLoadingStrategy, tensor_view::TensorReader,
    },
    stage::{ContiguousTilingLayout, StageMemory, TilingOrder},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::{AsyncLoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for LoadingStrategy<T> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let total_units = config.num_planes() * config.plane_dim();
        let num_slices = config.tiling_scheme().elements_in_tile_row(ident)
            * config.tiling_scheme().tiles_in_stage(ident);

        if num_slices >= total_units && num_slices % total_units != 0 {
            return Err(Box::new(format!(
                "Number of units ({total_units:?}) must divide number of slices ({num_slices:?}). Would require units doing different numbers of slices"
            )));
        }

        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> AsyncFullLoadingStrategy for LoadingStrategy<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job {
        let total_units = config.plane_dim() * config.num_planes();
        let line_size = config.global_line_size(input_ident);

        let (num_slices_per_tile, slice_length_in_lines) = match config.matrix_layout(input_ident) {
            MatrixLayout::RowMajor => (
                config.tiling_scheme().elements_in_tile_row(input_ident),
                config.tiling_scheme().elements_in_tile_col(input_ident) / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.tiling_scheme().elements_in_tile_col(input_ident),
                config.tiling_scheme().elements_in_tile_row(input_ident) / line_size,
            ),
        };

        let num_slices =
            comptime!(num_slices_per_tile * config.tiling_scheme().tiles_in_stage(input_ident));
        let num_tasks_per_unit = num_slices.div_ceil(total_units);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        Job {
            unit_id,
            num_tasks_per_unit,
            total_units,
            num_slices,
            input_ident,
            num_slices_per_tile,
            slice_length_in_lines,
            line_size,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    unit_id: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
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
    #[cube(comptime)]
    line_size: u32,
}

#[cube]
impl<MP: MatmulPrecision, TO: TilingOrder> AsyncLoadingJob<MP, ContiguousTilingLayout<TO>> for Job {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, ContiguousTilingLayout<TO>>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let slice_index = this.unit_id + this.total_units * task_id;

        let nth_tile = slice_index / this.num_slices_per_tile;
        let (tile_x, tile_y) = ContiguousTilingLayout::<TO>::to_x_y::<G::StageConfig>(
            nth_tile,
            comptime!(this.input_ident.as_ident()),
            config.stage_config(),
        );
        let nth_slice = slice_index % this.num_slices_per_tile;

        // TODO make branching comptime conditional (using balanced_workload)
        if slice_index < this.num_slices {
            let window = tensor_reader.load_window_in_tile::<G>(
                (tile_x, tile_y),
                nth_slice,
                this.input_ident,
                config,
            );

            // Where this unit writes source in the stage
            let slice_destination_offset =
                (nth_tile * this.num_slices_per_tile + nth_slice) * this.slice_length_in_lines;

            // Make destination start at offset
            let mut destination = stage.as_slice_mut(this.line_size).slice_mut(
                slice_destination_offset,
                slice_destination_offset + this.slice_length_in_lines,
            );

            CM::memcpy_async(
                mechanism,
                &window.slice.try_cast_unchecked(),
                &mut destination,
            );
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
