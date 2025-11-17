use crate::components::TilingScheme;
use crate::components::global::read::{FullLoadingStrategy, stage::FullStageLayout};
use crate::components::global::{GlobalReaderConfig, RoleRule};
use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::{StridedStage, StridedTilingLayout};
use crate::components::{InvalidConfigError, MatmulIdent};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the stage using all planes,
/// keeping the original layout, making each tile strided
pub struct SyncFullStridedLoading {}

impl LoadingValidation for SyncFullStridedLoading {
    fn check<R: Runtime>(
        _client: &ComputeClient<R::Server>,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        let line_size = config.gmem_config.line_size();

        let num_stage_lines = config.smem_config.elements_in_stage() / line_size;
        let total_units = config.loading_units_count();

        if !num_stage_lines.is_multiple_of(total_units) {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        StridedTilingLayout::check(config.smem_config)?;

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
impl FullLoadingStrategy for SyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = SyncFullStridedJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let num_stage_lines = config.smem_config.elements_in_stage() / line_size;
        let unit_count = config.loading_planes_count * config.plane_dim;
        let num_tasks_per_unit = comptime!(num_stage_lines / unit_count);

        let unit_position_base = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;

        SyncFullStridedJob {
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            line_size,
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
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, Synchronous>
    for SyncFullStridedJob
{
    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, StridedTilingLayout>,
        _barrier: &mut (),
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;

        let layout = FullStageLayout::new(comptime![config.smem_config]);
        let view = global_iter.view().view(layout);

        let line_read = view.read_checked(unit_position * this.line_size);

        stage.as_slice_mut(this.line_size)[unit_position] = Line::cast_from(line_read);
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
