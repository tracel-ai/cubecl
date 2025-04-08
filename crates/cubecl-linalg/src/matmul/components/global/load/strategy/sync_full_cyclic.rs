use std::marker::PhantomData;

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::global::load::strategy::base::default_sync_full_load;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, Stage, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::LoadingJob;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct SyncFullCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for SyncFullCyclicLoading<T> {
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
impl<T: TilingOrder> SyncFullLoadingStrategy for SyncFullCyclicLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;
    type Job<MP: MatmulPrecision> = SyncFullCyclicJob<MP, T>;

    fn load_full<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        default_sync_full_load::<Self, MP, G>(read_view, stage, quantization, input_ident, config)
    }

    fn job<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tiling = config.tiling_dimensions(input_ident);
        let tile_num_elements = tiling.tile_size();
        let line_size = config.global_line_size(input_ident);
        let num_stage_elements = tiling.total_size();
        let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
        let num_tasks = comptime!(num_stage_elements / jump_length);

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        SyncFullCyclicJob::<MP, T> {
            unit_position_base,
            read_view,
            stage,
            quantization,
            num_tasks,
            tile_num_elements,
            jump_length,
            line_size,
            input_ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncFullCyclicJob<MP: MatmulPrecision, T: TilingOrder> {
    unit_position_base: u32,

    read_view: TensorReader<MP::EI>,
    stage: Stage<MP::ES, ContiguousTilingLayout<T>>,
    quantization: CubeOption<Quantization<MP>>,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    tile_num_elements: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision, T: TilingOrder> LoadingJob<MP> for SyncFullCyclicJob<MP, T> {
    fn len(this: &Self) -> u32 {
        this.num_tasks.runtime()
    }

    fn execute_task<G: GlobalConfig>(this: &mut Self, task_id: u32, #[comptime] config: G) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        let nth_tile = unit_position / this.tile_num_elements;
        let pos_within_tile = unit_position % this.tile_num_elements;

        let (tile_x, tile_y) = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
            nth_tile,
            comptime!(this.input_ident.as_ident()),
            comptime!(config.to_smm_config()),
        );

        let line_read = this.read_view.load_coalesced_in_tile::<G>(
            tile_x,
            tile_y,
            pos_within_tile,
            this.input_ident,
            config,
        );

        this.stage.as_slice_mut()[unit_position / this.line_size] = match this.quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read),
            CubeOption::None => Line::cast_from(line_read),
        };
    }
}
