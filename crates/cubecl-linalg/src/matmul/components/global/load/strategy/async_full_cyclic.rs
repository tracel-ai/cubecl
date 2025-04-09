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

use super::{LoadingJob, LoadingJobConfig};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for LoadingStrategy<T> {
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
impl<TO: TilingOrder> AsyncFullLoadingStrategy for LoadingStrategy<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> = Job<MP, CM>;

    fn load_full<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        default_async_full_load::<Self, MP, G, CM>(
            tensor_reader,
            stage,
            mechanism,
            quantization,
            input_ident,
            config,
        )
    }

    fn new_job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job<MP, CM> {
        comptime! {
            if   quantization.is_some() {
                panic!("Quantization not supported on async loaders.")
            }
        }

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

        Job::<MP, CM> {
            unit_id,
            mechanism,
            job_config: comptime!(JobConfig {
                num_tasks,
                total_units,
                num_slices,
                input_ident,
                num_slices_per_tile,
                slice_length_in_lines,
            }),
            _phantom: PhantomData,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> {
    unit_id: u32,

    mechanism: CM,

    #[cube(comptime)]
    job_config: JobConfig,
    #[cube(comptime)]
    _phantom: PhantomData<MP>,
}

#[derive(Clone, Copy)]
pub struct JobConfig {
    num_tasks: u32,
    total_units: u32,
    num_slices: u32,
    input_ident: InputIdent,
    num_slices_per_tile: u32,
    slice_length_in_lines: u32,
}

impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, TO: TilingOrder>
    LoadingJobConfig<MP, ContiguousTilingLayout<TO>, Job<MP, CM>> for JobConfig
{
    fn len(job: &Job<MP, CM>) -> u32 {
        job.job_config.num_tasks
    }

    fn __expand_len(
        _context: &mut cubecl_core::prelude::Scope,
        job: <Job<MP, CM> as cubecl_core::prelude::CubeType>::ExpandType,
    ) -> u32 {
        job.job_config.num_tasks
    }
}

#[cube]
impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, TO: TilingOrder>
    LoadingJob<MP, ContiguousTilingLayout<TO>> for Job<MP, CM>
{
    type LoadingJobConfig = JobConfig;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, ContiguousTilingLayout<TO>>,
        #[comptime] config: G,
    ) {
        let jc = this.job_config;

        let slice_index = this.unit_id + jc.total_units * task_id;

        let nth_tile = slice_index / jc.num_slices_per_tile;
        let (tile_x, tile_y) = ContiguousTilingLayout::<TO>::to_x_y::<G::SmmConfig>(
            nth_tile,
            comptime!(jc.input_ident.as_ident()),
            config.to_smm_config(),
        );
        let nth_slice = slice_index % jc.num_slices_per_tile;

        // TODO make branching comptime conditional
        if slice_index < jc.num_slices {
            let window = tensor_reader.load_window_in_tile::<G>(
                (tile_x, tile_y),
                nth_slice,
                jc.input_ident,
                config,
            );

            // Where this unit writes source in the stage
            let slice_destination_offset =
                (nth_tile * jc.num_slices_per_tile + nth_slice) * jc.slice_length_in_lines;

            // Make destination start at offset
            let mut destination = stage.as_slice_mut().slice_mut(
                slice_destination_offset,
                slice_destination_offset + jc.slice_length_in_lines,
            );

            CM::memcpy_async(
                &this.mechanism,
                &window.slice.try_cast_unchecked(),
                &mut destination,
            );
        }
    }
}
