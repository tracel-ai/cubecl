use crate::components::TilingScheme;
use crate::components::global::read::GlobalReaderConfig;
use crate::components::global::read::{FullLoadingStrategy, stage::FullStageLayout};
use crate::components::global::{GlobalConfig, RoleRule};
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
    fn check<C: GlobalReaderConfig, R: Runtime>(
        _client: &ComputeClient<R::Server>,
        config: &C,
        ident: MatmulIdent,
    ) -> Result<(), InvalidConfigError> {
        let line_size = config.global_line_size(ident);

        let num_stage_lines = config.tiling_scheme().elements_in_stage(ident) / line_size;
        let total_units = config.num_loading_planes(ident) * config.plane_dim();

        if !num_stage_lines.is_multiple_of(total_units) {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        StridedTilingLayout::check(config.global_memory_config(ident))?;

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

    fn new_job<EG: Numeric, ES: Numeric, G: GlobalReaderConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] line_size: u32,
        #[comptime] config: G,
    ) -> Self::Job<EG, ES> {
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
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, Synchronous>
    for SyncFullStridedJob
{
    fn execute_task<G: GlobalReaderConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, StridedTilingLayout>,
        _barrier: &mut (),
        #[comptime] config: G,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;

        let layout = FullStageLayout::new(comptime![config.global_memory_config(this.ident)]);
        let view = global_iter.view().view(layout);

        let line_read = view.read_checked(unit_position * this.line_size);

        stage.as_slice_mut(this.line_size)[unit_position] = Line::cast_from(line_read);
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
