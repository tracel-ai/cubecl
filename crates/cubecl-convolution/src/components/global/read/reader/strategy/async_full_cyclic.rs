use std::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_matmul::components::{
    InvalidConfigError, MatmulElems, MatmulProblem,
    global::{
        GlobalReaderConfig, RoleRule,
        memory::GlobalIterator,
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            LoadingJob, LoadingValidation, ReaderMode, async_barrier::AsyncCopy,
            async_full_cyclic::AsyncFullCyclicLoading as MatmulCyclicLoading, tiled::TiledLayout,
        },
    },
    stage::{ContiguousTilingLayout, StridedStageFamily, StridedStageMemory, TilingOrder},
};
use cubecl_std::tensor::layout::{Layout, LayoutExpand};

use crate::components::global::{
    args::RuntimeArgs,
    read::{
        full_reader::FullLoadingStrategy,
        strategy::async_copy::{ASYNC_COPY_WIDTH, async_copy_from},
    },
};

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
        MatmulCyclicLoading::<TO>::check(client, problem, config, dtypes)
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncFullCyclicLoading<TO> {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: u8,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        MatmulCyclicLoading::<TO>::max_round_plane_count(
            elements_per_tile,
            tiles_per_stage,
            line_size,
            plane_dim,
            dtype,
        )
    }
}

#[cube]
impl<TO: TilingOrder> FullLoadingStrategy for AsyncFullCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullCyclicJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        runtime_args: RuntimeArgs,
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
            runtime_args,
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

#[derive(CubeType, Clone)]
pub struct AsyncFullCyclicJob {
    unit_position_base: u32,
    runtime_args: RuntimeArgs,

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
            copy_line::<EG, ES, TO>(
                this,
                unit_position,
                global_iter,
                stage,
                &this.runtime_args,
                config,
            );
        } else {
            if unit_position < this.num_stage_elements {
                copy_line::<EG, ES, TO>(
                    this,
                    unit_position,
                    global_iter,
                    stage,
                    &this.runtime_args,
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
    job: &AsyncFullCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Line<EG>>,
    stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
    runtime_args: &RuntimeArgs,
    #[comptime] config: GlobalReaderConfig,
) {
    let nth_tile = unit_position / job.tile_num_elements;
    let pos_within_tile = unit_position % job.tile_num_elements;

    let layout = TiledLayout::new(config.stage_ident, config.smem_config);
    let view = global_iter.view();

    let tile = ContiguousTilingLayout::<TO>::to_x_y(nth_tile, config.smem_config);

    let pos = layout.to_source_pos((tile, pos_within_tile));
    let stage_offset = unit_position / stage.smem.line_size();

    async_copy_from(
        view,
        pos,
        stage,
        stage_offset,
        runtime_args,
        global_iter.offset(),
        config,
        job.copy_line_size,
    );
}
