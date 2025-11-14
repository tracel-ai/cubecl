use std::marker::PhantomData;

use crate::components::TilingScheme;
use crate::components::global::read::{FullLoadingStrategy, tiled::TiledLayout};
use crate::components::global::{GlobalReaderConfig, RoleRule};
use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::{ContiguousTilingLayout, StridedStage, TilingOrder};
use crate::components::{InvalidConfigError, MatmulIdent};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{LoadingJob, LoadingValidation, ReaderMode};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct SyncFullCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _t: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for SyncFullCyclicLoading<TO> {
    fn check<R: Runtime>(
        _client: &ComputeClient<R::Server>,
        config: &GlobalReaderConfig,
        ident: MatmulIdent,
    ) -> Result<(), InvalidConfigError> {
        if let ReaderMode::Strict = config.reader_mode() {
            let line_size = config.global_line_size(ident);

            let num_stage_lines = config.tiling_scheme().elements_in_stage(ident) / line_size;
            let total_units = config.num_loading_planes(ident) * config.plane_dim();

            if !num_stage_lines.is_multiple_of(total_units) {
                return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
            }
        }

        ContiguousTilingLayout::<TO>::check(config.global_memory_config(ident))?;

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for SyncFullCyclicLoading<TO> {
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
impl<TO: TilingOrder> FullLoadingStrategy for SyncFullCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = SyncFullCyclicJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] ident: MatmulIdent,
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let tile_num_elements = config.tiling_scheme().elements_in_tile(ident);
        let num_stage_elements = config.tiling_scheme().elements_in_stage(ident);

        let num_stage_lines = num_stage_elements.div_ceil(line_size);
        let total_units = comptime!(config.num_loading_planes(ident) * config.plane_dim());
        let num_tasks_per_unit = comptime!(num_stage_lines.div_ceil(total_units));
        let balanced_workload = comptime!(num_stage_lines.is_multiple_of(total_units));
        let jump_length = comptime!(total_units * line_size);

        let unit_id = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides())
            * config.plane_dim()
            + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        SyncFullCyclicJob {
            unit_position_base,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            line_size,
            ident,
            balanced_workload,
            num_stage_elements,
            reader_mode: comptime!(config.reader_mode()),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncFullCyclicJob {
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
    ident: MatmulIdent,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, Synchronous> for SyncFullCyclicJob
{
    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, ContiguousTilingLayout<TO>>,
        _barrier: &mut (),
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            load_and_store_line::<EG, ES, TO>(this, unit_position, global_iter, stage, config);
        } else {
            if unit_position < this.num_stage_elements {
                load_and_store_line::<EG, ES, TO>(this, unit_position, global_iter, stage, config);
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn load_and_store_line<EG: Numeric, ES: Numeric, TO: TilingOrder>(
    job: &SyncFullCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut StridedStage<ES, ContiguousTilingLayout<TO>>,
    #[comptime] config: GlobalReaderConfig,
) {
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;

    let layout = TiledLayout::new(comptime![config.global_memory_config(job.ident)]);
    let view = global_iter.view().view(layout);

    let tile = ContiguousTilingLayout::<TO>::to_x_y(
        nth_tile,
        comptime!(config.stage_memory_config(job.ident)),
    );

    let line_read = view.read_checked((tile, pos_within_tile));

    stage.as_slice_mut(job.line_size)[unit_position / job.line_size] = Line::cast_from(line_read);
}
