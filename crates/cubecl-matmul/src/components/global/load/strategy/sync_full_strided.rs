use crate::components::global::load::SyncFullLoadingStrategy;
use crate::components::global::memory::TensorReader;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::{GlobalConfig, Quantization, RoleRule};
use crate::components::stage::{StageMemory, StridedTilingLayout};
use crate::components::{InvalidConfigError, MatmulIdent};
use crate::components::{MatmulPrecision, TilingScheme};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the stage using all planes,
/// keeping the original layout, making each tile strided
pub struct SyncFullStridedLoading {}

impl LoadingValidation for SyncFullStridedLoading {
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        let line_size = config.global_line_size(ident);

        let num_stage_lines = config.tiling_scheme().elements_in_stage(ident) / line_size;
        let total_units = config.num_loading_planes(ident) * config.plane_dim();

        if num_stage_lines % total_units != 0 {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for SyncFullStridedLoading {
    fn max_round_plane_count(
        tiling_scheme: &TilingScheme,
        ident: MatmulIdent,
        line_size: u8,
        plane_dim: u32,
    ) -> u32 {
        let num_lines = tiling_scheme.elements_in_stage(ident) / line_size as u32;
        num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl SyncFullLoadingStrategy for SyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = SyncFullStridedJob;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let line_size = config.global_line_size(ident);
        let num_stage_lines = config.tiling_scheme().elements_in_stage(ident) / line_size;
        let unit_count = config.num_loading_planes(ident) * config.plane_dim();
        let num_tasks_per_unit = comptime!(num_stage_lines / unit_count);

        let unit_position_base = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides())
            * config.plane_dim()
            + UNIT_POS_X;

        SyncFullStridedJob {
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            line_size,
            ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncFullStridedJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<MP: MatmulPrecision> LoadingJob<MP, StridedTilingLayout> for SyncFullStridedJob {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, StridedTilingLayout>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;

        let line_read = tensor_reader.load_coalesced_in_stage(
            unit_position * this.line_size,
            comptime!(config.global_memory_config(this.ident)),
        );

        stage.as_slice_mut(this.line_size)[unit_position] = match quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read, this.ident),
            CubeOption::None => Line::cast_from(line_read),
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
