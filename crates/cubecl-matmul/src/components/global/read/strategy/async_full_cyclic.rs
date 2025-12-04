use std::marker::PhantomData;

use crate::components::global::read::{
    FullLoadingStrategy, async_barrier::AsyncCopy, async_copy::ASYNC_COPY_WIDTH, tiled::TiledLayout,
};
use crate::components::global::read::{validate_async_barrier, validate_swizzle_atom_size};
use crate::components::global::{GlobalReaderConfig, RoleRule};
use crate::components::global::{
    multi_stage::LoadMaxRoundPlaneCount, read::async_copy::async_copy_from,
};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{ContiguousTilingLayout, StridedStageMemory, TilingOrder};
use crate::components::{InvalidConfigError, MatmulProblem};
use crate::components::{MatmulElems, global::read::validate_async_copy};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::tensor::layout::{Layout, LayoutExpand};

use super::{LoadingJob, LoadingValidation, ReaderMode};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct AsyncFullCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _t: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for AsyncFullCyclicLoading<TO> {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        let line_size =
            ASYNC_COPY_WIDTH / dtypes.stage(config.stage_ident.into()).size_bits() as u32;

        if let ReaderMode::Strict = config.reader_mode {
            let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
            let total_units = config.loading_units_count();

            if !num_stage_lines.is_multiple_of(total_units) {
                return Err(Box::new(format!(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {total_units:?} divides number of lines in stage.",
            )));
            }
        }

        // Needs separate check because copy size may be larger than global line size
        if !config
            .smem_config
            .elements_per_tile_along_contiguous_dim()
            .is_multiple_of(line_size)
        {
            return Err(Box::new("Tile size isn't divisible by copy line size"));
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        validate_async_copy(client, problem, dtypes, config)?;
        validate_async_barrier(client)?;
        ContiguousTilingLayout::<TO>::check(config.smem_config)?;

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncFullCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        _line_size: u8,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        let line_size = ASYNC_COPY_WIDTH / dtype.size_bits() as u32;
        let elements_per_stage = elements_per_tile * tiles_per_stage;
        let num_lines = elements_per_stage / line_size;
        num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl<TO: TilingOrder> FullLoadingStrategy for AsyncFullCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullCyclicJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] _line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let type_size = ES::type_size_bits();
        let line_size = comptime![ASYNC_COPY_WIDTH / type_size];
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

        AsyncFullCyclicJob {
            unit_position_base,
            num_tasks_per_unit,
            tile_num_elements,
            jump_length,
            copy_line_size: line_size,
            balanced_workload,
            num_stage_elements,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullCyclicJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    tile_num_elements: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    copy_line_size: u32,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, AsyncCopy> for AsyncFullCyclicJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
        _barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            copy_line::<EG, ES, TO>(this, unit_position, global_iter, stage, config);
        } else {
            if unit_position < this.num_stage_elements {
                copy_line::<EG, ES, TO>(this, unit_position, global_iter, stage, config);
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn copy_line<EG: Numeric, ES: Numeric, TO: TilingOrder>(
    job: &AsyncFullCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
    #[comptime] config: GlobalReaderConfig,
) {
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;

    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view();

    let tile = ContiguousTilingLayout::<TO>::to_x_y(nth_tile, config.smem_config);

    let pos = layout.to_source_pos((tile, pos_within_tile));
    let stage_offset = unit_position / stage.smem.line_size();

    async_copy_from(view, pos, stage, stage_offset, config, job.copy_line_size);
}
