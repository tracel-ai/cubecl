use std::marker::PhantomData;

use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::{SyncFullLoadingStrategy, default_sync_full_load};
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

use super::{LoadingJob, LoadingJobConfig};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using
/// one plane per tile.
pub struct SyncFullTilewiseLoading<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for SyncFullTilewiseLoading<T> {
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
impl<T: TilingOrder> SyncFullLoadingStrategy for SyncFullTilewiseLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;
    type Job<MP: MatmulPrecision> = Job<MP, T>;

    fn load_full<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        default_sync_full_load::<Self, MP, G>(read_view, stage, quantization, input_ident, config)
    }

    fn job<MP: MatmulPrecision, G: GlobalConfig>(
        stage: Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tiling = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);

        let num_lines_per_tile = comptime!(tiling.tile_size() / line_size);

        let nth_tile = UNIT_POS_Y;
        let offset_base = num_lines_per_tile * nth_tile;

        let num_tasks = num_lines_per_tile / config.plane_dim();

        let tile = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
            nth_tile,
            input_ident.as_ident(),
            config.to_smm_config(),
        );

        Job::<MP, T> {
            tile,
            offset_base,
            stage,
            quantization,
            job_config: comptime!(JobConfig {
                num_tasks,
                line_size,
                input_ident,
            }),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
struct Job<MP: MatmulPrecision, T: TilingOrder> {
    tile: (u32, u32),
    offset_base: u32,

    stage: Stage<MP::ES, ContiguousTilingLayout<T>>,
    quantization: CubeOption<Quantization<MP>>,

    #[cube(comptime)]
    job_config: JobConfig,
}

#[derive(Copy, Clone)]
struct JobConfig {
    num_tasks: u32,
    line_size: u32,
    input_ident: InputIdent,
}

impl<MP: MatmulPrecision, T: TilingOrder> LoadingJobConfig<MP, Job<MP, T>> for JobConfig {
    fn len(job: &Job<MP, T>) -> u32 {
        job.job_config.num_tasks
    }

    fn __expand_len(
        _context: &mut cubecl_core::prelude::Scope,
        job: <Job<MP, T> as cubecl_core::prelude::CubeType>::ExpandType,
    ) -> u32 {
        job.job_config.num_tasks
    }
}

#[cube]
impl<MP: MatmulPrecision, T: TilingOrder> LoadingJob<MP> for Job<MP, T> {
    type LoadingJobConfig = JobConfig;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        read_view: &TensorReader<MP::EI>,
        #[comptime] config: G,
    ) {
        let pos_within_tile = task_id * comptime!(config.plane_dim()) + UNIT_POS_X;

        let line_read = read_view.load_coalesced_in_tile::<G>(
            this.tile.0,
            this.tile.1,
            pos_within_tile * this.job_config.line_size,
            this.job_config.input_ident,
            config,
        );

        let offset = this.offset_base + pos_within_tile;

        this.stage.as_slice_mut()[offset] = match this.quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read),
            CubeOption::None => Line::cast_from(line_read),
        }
    }
}
