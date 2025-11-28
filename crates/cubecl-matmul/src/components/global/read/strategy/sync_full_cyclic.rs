use std::marker::PhantomData;

use crate::components::MatmulElems;
use crate::components::global::read::validate_swizzle_atom_size;
use crate::components::global::read::{FullLoadingStrategy, tiled::TiledLayout};
use crate::components::global::{GlobalReaderConfig, RoleRule};
use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{ContiguousTilingLayout, StridedStageMemory, TilingOrder};
use crate::components::{InvalidConfigError, MatmulProblem};
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
        _client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        if let ReaderMode::Strict = config.reader_mode {
            let line_size = config.gmem_config.line_size;

            let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
            let total_units = config.loading_units_count();

            if !num_stage_lines.is_multiple_of(total_units) {
                return Err(Box::new(format!(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {total_units:?} divides number of lines in stage.",
            )));
            }
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        ContiguousTilingLayout::<TO>::check(config.smem_config)?;

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for SyncFullCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: u8,
        plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        let elements_per_stage = elements_per_tile * tiles_per_stage;
        let num_lines = elements_per_stage / line_size as u32;
        num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl<TO: TilingOrder> FullLoadingStrategy for SyncFullCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = SyncFullCyclicJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let tile_num_elements = config.smem_config.elements_per_tile();
        let num_stage_elements = config.smem_config.elements_per_stage();

        let num_stage_lines = num_stage_elements.div_ceil(line_size);
        let total_units = config.loading_units_count();
        let num_tasks_per_unit = comptime!(num_stage_lines.div_ceil(total_units));
        let balanced_workload = comptime!(num_stage_lines.is_multiple_of(total_units));
        let jump_length = comptime!(total_units * line_size);

        let unit_id = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        SyncFullCyclicJob {
            unit_position_base,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            line_size,
            balanced_workload,
            num_stage_elements,
            reader_mode: config.reader_mode,
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
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
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
    stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
    #[comptime] config: GlobalReaderConfig,
) {
    let line_size = job.line_size;
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;

    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view().view(layout);

    let tile = ContiguousTilingLayout::<TO>::to_x_y(nth_tile, comptime!(config.smem_config));

    let mut slice = stage.as_slice_mut(line_size);

    let line_read = view.read_checked((tile, pos_within_tile));
    let stage_offs = stage.swizzle.apply(unit_position, ES::type_size());

    slice[stage_offs / job.line_size] = Line::cast_from(line_read);
}
