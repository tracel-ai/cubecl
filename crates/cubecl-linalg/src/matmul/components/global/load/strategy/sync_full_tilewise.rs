use std::marker::PhantomData;

use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::{
    FormattedConfigError, Ident, InputIdent, InvalidConfigError, MatmulPrecision,
};
use crate::matmul::components::{
    global::{GlobalConfig, LoadingValidation, tensor_view::TensorReader},
    stage::{ContiguousTilingLayout, Stage, TilingOrder},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::LoadingJob;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using
/// one plane per tile.
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for LoadingStrategy<T> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_planes = config.num_planes();
        let num_tiles = tiling.tile_count();

        if num_planes != num_tiles {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {:?} must equal number of tiles {:?} for tilewise loading.",
                    num_planes, num_tiles,
                )
            }));
        }

        if line_size != config.stage_line_size(ident) {
            return Err(Box::new(
                "Global and stage line sizes must match for tilewise loading.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl<T: TilingOrder> SyncFullLoadingStrategy for LoadingStrategy<T> {
    type TilingLayout = ContiguousTilingLayout<T>;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tiling = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);

        let num_lines_per_tile = comptime!(tiling.tile_size() / line_size);

        let nth_tile = UNIT_POS_Y;
        let previous_tiles_offset = num_lines_per_tile * nth_tile;

        let tile = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
            nth_tile,
            input_ident.as_ident(),
            config.to_smm_config(),
        );

        Job {
            tile,
            previous_tiles_offset,
            num_tasks: num_lines_per_tile,
            num_workers: config.plane_dim(),
            line_size,
            input_ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    tile: (u32, u32),
    previous_tiles_offset: u32,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    num_workers: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
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
        let pos_within_tile = task_id * comptime!(config.plane_dim()) + UNIT_POS_X;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_tasks % this.num_workers == 0) {
            Job::load_and_store_line::<MP, TO, G>(
                this,
                pos_within_tile,
                tensor_reader,
                stage,
                quantization,
                config,
            );
        } else {
            if pos_within_tile * this.line_size
                < comptime!(config.tiling_dimensions(this.input_ident).tile_size())
            {
                Job::load_and_store_line::<MP, TO, G>(
                    this,
                    pos_within_tile,
                    tensor_reader,
                    stage,
                    quantization,
                    config,
                );
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        comptime!(this.num_tasks.div_ceil(this.num_workers))
    }
}

#[cube]
impl Job {
    fn load_and_store_line<MP: MatmulPrecision, TO: TilingOrder, G: GlobalConfig>(
        this: &Self,
        pos_within_tile: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, ContiguousTilingLayout<TO>>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let line_read = tensor_reader.load_coalesced_in_tile::<G>(
            this.tile.0,
            this.tile.1,
            pos_within_tile * this.line_size,
            this.input_ident,
            config,
        );

        let offset = this.previous_tiles_offset + pos_within_tile;

        stage.as_slice_mut()[offset] = match quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read),
            CubeOption::None => Line::cast_from(line_read),
        };
    }
}
