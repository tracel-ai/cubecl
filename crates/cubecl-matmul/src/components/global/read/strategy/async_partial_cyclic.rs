use std::marker::PhantomData;

use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::read::{PartialLoadingStrategy, tiled::TiledLayout};
use crate::components::global::read::{async_barrier::AsyncBarrier, validate_swizzle_atom_size};
use crate::components::global::{GlobalReaderConfig, RoleRule};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::StridedStageMemory;
use crate::components::stage::{ContiguousTilingLayout, TilingOrder};
use crate::components::{InvalidConfigError, StageIdent};
use crate::components::{MatmulElems, global::read::validate_async_barrier};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::type_size;

use super::{LoadingJob, LoadingValidation, ReaderMode};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct AsyncPartialCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for AsyncPartialCyclicLoading<TO> {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        if let ReaderMode::Strict = config.reader_mode {
            let line_size = config.gmem_config.line_size;
            let num_lines_per_tile = config.smem_config.elements_per_tile() / line_size;
            let num_tiles_in_stage = config.smem_config.tiles_per_stage();
            let total_num_lines = num_tiles_in_stage * num_lines_per_tile;

            let total_units = config.loading_units_count();
            let jump_length = total_units * line_size;
            let num_tasks_per_unit = total_num_lines.div_ceil(total_units);

            let max_id = total_units - 1;
            let max_task_id = num_tasks_per_unit - 1;
            let max_position_base = max_id * line_size;
            let max_position = max_position_base + max_task_id * jump_length;
            let num_stage_elements = config.smem_config.elements_per_stage();

            if max_position > num_stage_elements {
                return Err(Box::new(
                    "Too many data will be loaded, resulting in out-of-bounds",
                ));
            }
        }

        if dtypes.stage(config.stage_ident.into()).size()
            != dtypes.global(config.stage_ident.into()).size()
        {
            return Err(Box::new(
                "Memcpy can't cast in flight, so stage and global must be the same",
            ));
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        validate_async_barrier(client)?;
        ContiguousTilingLayout::<TO>::check(config.smem_config)?;

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncPartialCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: u8,
        plane_dim: u32,
    ) -> u32 {
        let num_lines_per_tile = elements_per_tile / line_size as u32;
        let total_num_lines = tiles_per_stage * num_lines_per_tile;
        total_num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl<TO: TilingOrder> PartialLoadingStrategy for AsyncPartialCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncBarrier;
    type Stage = StridedStageFamily;

    type Job<EG: Numeric, ES: Numeric> = AsyncPartialCyclicJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncPartialCyclicJob {
        let num_stage_elements = config.smem_config.elements_per_stage();

        let tile_size = config.smem_config.elements_per_tile();
        let tile_count_row = config.smem_config.tiles_per_stage_along_row();
        let tile_count_col = config.smem_config.tiles_per_stage_along_col();

        let num_lines_per_tile = tile_size / line_size;
        let total_units = config.loading_units_count();

        let num_tiles_in_stage = tile_count_row * tile_count_col;
        let total_num_lines = num_tiles_in_stage * num_lines_per_tile;
        let balanced_workload = total_num_lines.is_multiple_of(total_units);
        let num_tasks_per_unit = total_num_lines.div_ceil(total_units);
        let jump_length = total_units * line_size;

        let plane_id = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config);
        let unit_id = plane_id * config.plane_dim + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        AsyncPartialCyclicJob {
            unit_position_base,
            num_tasks_per_unit,
            stage_index,
            jump_length,
            num_lines_per_tile,
            balanced_workload,
            num_stage_elements,
            reader_mode: config.reader_mode,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialCyclicJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    num_lines_per_tile: u32,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, AsyncBarrier> for AsyncPartialCyclicJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
        barrier: &mut Barrier,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;
        let mut stage = stage.with_buffer_index(this.stage_index);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            copy_line::<EG, ES, TO>(
                this,
                unit_position,
                global_iter,
                &mut stage,
                barrier,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                copy_line::<EG, ES, TO>(
                    this,
                    unit_position,
                    global_iter,
                    &mut stage,
                    barrier,
                    config,
                );
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn copy_line<EG: Numeric, ES: Numeric, TO: TilingOrder>(
    job: &AsyncPartialCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
    barrier: &mut Barrier,
    #[comptime] config: GlobalReaderConfig,
) {
    let layout = TiledLayout::new(comptime!(config.smem_config));
    let view = global_iter.view().view(layout);

    let (tile_size, tile_count_row, tile_count_col) = comptime! {
        (
            config.smem_config.elements_per_tile(),
            config.smem_config.tiles_per_stage_along_row(),
            config.smem_config.tiles_per_stage_along_col(),
        )
    };
    let line_size = view.line_size();

    let tile_index = unit_position / tile_size;
    let pos_within_tile = unit_position % tile_size;

    let (tile_x_within_stage, tile_y_within_stage) = TO::to_row_col(
        tile_index,
        tile_count_row,
        tile_count_col,
        comptime!(config.smem_config),
    );

    let tile = match comptime!(config.stage_ident) {
        StageIdent::Lhs => (
            tile_x_within_stage,
            job.stage_index * tile_count_col + tile_y_within_stage,
        ),
        StageIdent::Rhs => (
            job.stage_index * tile_count_row + tile_x_within_stage,
            tile_y_within_stage,
        ),
        _ => comptime!(unreachable!()),
    };

    let mut slice = stage.as_slice_mut(line_size);
    let global_slice =
        view.slice_unchecked((tile, pos_within_tile), ((1u32, 1u32), 1u32).runtime());

    let tile_start = tile_index * job.num_lines_per_tile;
    let offset = tile_start + pos_within_tile / line_size;
    let type_size = type_size::<ES>(line_size);
    let offset = stage.swizzle.apply(offset, type_size);

    let stage_slice = slice.slice_mut(offset, offset + 1);

    barrier.memcpy_async(
        &global_slice.to_linear_slice(),
        &mut stage_slice.try_cast_unchecked(),
    );
}
