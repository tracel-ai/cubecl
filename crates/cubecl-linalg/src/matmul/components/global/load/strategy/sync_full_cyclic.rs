use std::marker::PhantomData;

use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, Stage, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use crate::matmul::components::{MatmulPrecision, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::LoadingJob;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for LoadingStrategy<TO> {
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

        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> SyncFullLoadingStrategy for LoadingStrategy<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tiling = config.tiling_dimensions(input_ident);
        let tile_num_elements = tiling.tile_size();
        let line_size = config.global_line_size(input_ident);
        let num_stage_elements = tiling.total_size();
        let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
        let num_tasks_per_unit = comptime!(num_stage_elements / jump_length);
        let segment_length = match config.matrix_layout(input_ident) {
            MatrixLayout::RowMajor => tiling.tile_shape_col(),
            MatrixLayout::ColMajor => tiling.tile_shape_row(),
        };

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        Job {
            unit_position_base,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            segment_length,
            line_size,
            input_ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    pub unit_position_base: u32,

    #[cube(comptime)]
    pub num_tasks_per_unit: u32,
    #[cube(comptime)]
    pub tile_num_elements: u32,
    #[cube(comptime)]
    pub jump_length: u32,
    #[cube(comptime)]
    pub segment_length: u32,
    #[cube(comptime)]
    pub line_size: u32,
    #[cube(comptime)]
    pub input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision, TO: TilingOrder> LoadingJob<MP, ContiguousTilingLayout<TO>> for Job {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, ContiguousTilingLayout<TO>>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        load_and_store_line::<MP, TO, G>(
            this,
            unit_position,
            tensor_reader,
            stage,
            quantization,
            config,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn load_and_store_line<MP: MatmulPrecision, TO: TilingOrder, G: GlobalConfig>(
    job: &Job,
    unit_position: u32,
    tensor_reader: &TensorReader<MP::EI>,
    stage: &mut Stage<MP::ES, ContiguousTilingLayout<TO>>,
    quantization: &CubeOption<Quantization<MP>>,
    #[comptime] config: G,
) {
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;
    let segment_index = pos_within_tile / job.segment_length;
    let pos_in_segment = pos_within_tile % job.segment_length;

    let (tile_x, tile_y) = ContiguousTilingLayout::<TO>::to_x_y::<G::SmmConfig>(
        nth_tile,
        comptime!(job.input_ident.as_ident()),
        comptime!(config.to_smm_config()),
    );

    let line_read = tensor_reader.load_coalesced_in_tile::<G>(
        tile_x,
        tile_y,
        segment_index,
        pos_in_segment,
        job.input_ident,
        config,
    );

    let line_read = match quantization {
        CubeOption::Some(quantization) => quantization.dequantize(line_read, job.input_ident),
        CubeOption::None => Line::cast_from(line_read),
    };

    let mut segment_slice = stage
        .segment::<G::SmmConfig>(
            tile_x,
            tile_y,
            segment_index,
            job.input_ident,
            config.to_smm_config(),
        )
        .as_data_slice_mut(job.line_size);
    segment_slice[pos_in_segment / job.line_size] = line_read;
}
