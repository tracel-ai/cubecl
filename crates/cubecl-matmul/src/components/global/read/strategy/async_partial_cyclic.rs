use std::marker::PhantomData;

use crate::components::global::{
    multi_stage::LoadMaxRoundPlaneCount,
    read::{
        AsyncPartialLoadingStrategy, async_barrier::AsyncCopy, async_copy::async_copy_from,
        validate_async_copy,
    },
};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::StridedStageMemory;
use crate::components::stage::{ContiguousTilingLayout, TilingOrder};
use crate::components::{InvalidConfigError, StageIdent};
use crate::components::{MatmulElems, global::read::validate_async_barrier};
use crate::components::{MatmulPrecision, global::read::validate_swizzle_atom_size};
use crate::components::{
    MatmulProblem,
    global::{GlobalReaderConfig, RoleRule, read::async_copy::ASYNC_COPY_WIDTH},
};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::components::{
    global::{
        SharedGlobalMatmulConfig,
        read::{PartialLoadingStrategy, tiled::TiledLayout},
    },
    stage::StageConfig,
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::tensor::layout::{Layout, LayoutExpand};

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
        problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        let line_size =
            ASYNC_COPY_WIDTH / dtypes.stage(config.stage_ident.into()).size_bits() as u32;
        if let ReaderMode::Strict = config.reader_mode {
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

        // Needs separate check because copy size may be larger than global line size
        if !config
            .smem_config
            .elements_per_tile_along_contiguous_dim()
            .is_multiple_of(line_size)
        {
            return Err(Box::new("Tile size isn't divisible by copy line size"));
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        validate_async_barrier(client)?;
        validate_async_copy(client, problem, dtypes, config)?;
        ContiguousTilingLayout::<TO>::check(config.smem_config)?;

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncPartialCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        _line_size: u8,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        let line_size = ASYNC_COPY_WIDTH / dtype.size_bits() as u32;
        let num_lines_per_tile = elements_per_tile / line_size;
        let total_num_lines = tiles_per_stage * num_lines_per_tile;
        total_num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl<TO: TilingOrder> PartialLoadingStrategy for AsyncPartialCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncCopy;
    type Stage = StridedStageFamily;

    type Job<EG: Numeric, ES: Numeric> = AsyncPartialCyclicJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
        #[comptime] _line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncPartialCyclicJob {
        let type_size = ES::type_size_bits();
        let line_size = comptime![ASYNC_COPY_WIDTH / type_size];
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
            copy_line_size: line_size,
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
    #[cube(comptime)]
    copy_line_size: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, AsyncCopy> for AsyncPartialCyclicJob
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
        let mut stage = stage.with_buffer_index(this.stage_index);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            copy_line::<EG, ES, TO>(this, unit_position, global_iter, &mut stage, config);
        } else {
            if unit_position < this.num_stage_elements {
                copy_line::<EG, ES, TO>(this, unit_position, global_iter, &mut stage, config);
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
    #[comptime] config: GlobalReaderConfig,
) {
    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view();

    let (tile_size, tile_count_row, tile_count_col) = comptime! {
        (
            config.smem_config.elements_per_tile(),
            config.smem_config.tiles_per_stage_along_row(),
            config.smem_config.tiles_per_stage_along_col(),
        )
    };

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

    let pos = layout.to_source_pos((tile, pos_within_tile));

    let tile_start = tile_index * job.num_lines_per_tile * job.copy_line_size;
    let stage_offset = (tile_start + pos_within_tile) / stage.smem.line_size();

    async_copy_from(view, pos, stage, stage_offset, config, job.copy_line_size);
}

#[cube]
impl<TO: TilingOrder> AsyncPartialLoadingStrategy for AsyncPartialCyclicLoading<TO> {
    fn arrival_count<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> u32 {
        let total_load_units =
            config.plane_role_config().plane_roles.load_only * config.plane_dim();
        total_load_units.runtime()
    }

    fn barrier_post_init() {}

    fn arrive<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.commit_copy_async();
        barrier.arrive();
    }

    fn is_elected<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> bool {
        let role_rule = RoleRule::new(config.plane_role_config().rule);
        role_rule.is_load_plane()
    }
}
