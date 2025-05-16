use std::marker::PhantomData;

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::load::{SyncFullLoadingStrategy, sync_full_cyclic};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, Quantization};
use crate::matmul::components::stage::{ContiguousTilingLayout, StageMemory, TilingOrder};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Same as sync_full_cyclic but the guard is runtime
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for LoadingStrategy<TO> {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
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
        let num_stage_lines = num_stage_elements / line_size;
        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_tasks_per_unit = comptime!(num_stage_lines.div_ceil(total_units));
        let balanced_workload = num_tasks_per_unit % total_units == 0;

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        Job {
            unit_position_base,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            line_size,
            input_ident,
            balanced_workload,
            num_stage_elements,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    tile_num_elements: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
}

#[cube]
impl<MP: MatmulPrecision, TO: TilingOrder> LoadingJob<MP, ContiguousTilingLayout<TO>> for Job {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, ContiguousTilingLayout<TO>>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        let cyclic_job = sync_full_cyclic::Job {
            unit_position_base: this.unit_position_base,
            num_tasks_per_unit: comptime!(this.num_tasks_per_unit),
            tile_num_elements: comptime!(this.tile_num_elements),
            jump_length: comptime!(this.jump_length),
            line_size: comptime!(this.line_size),
            input_ident: comptime!(this.input_ident),
        };

        #[allow(clippy::collapsible_else_if)]
        if this.balanced_workload {
            sync_full_cyclic::load_and_store_line::<MP, TO, G>(
                &cyclic_job,
                unit_position,
                tensor_reader,
                stage,
                quantization,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                sync_full_cyclic::load_and_store_line::<MP, TO, G>(
                    &cyclic_job,
                    unit_position,
                    tensor_reader,
                    stage,
                    quantization,
                    config,
                );
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
