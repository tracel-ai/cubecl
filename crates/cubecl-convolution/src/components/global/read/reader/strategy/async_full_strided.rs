use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_matmul::components::{
    InvalidConfigError, MatmulElems, MatmulProblem,
    global::{
        GlobalReaderConfig, RoleRule,
        memory::GlobalIterator,
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            LoadingJob, LoadingValidation, async_barrier::AsyncCopy,
            async_full_strided::AsyncFullStridedLoading as MatmulStridedLoading,
            stage::FullStageLayout,
        },
    },
    stage::{StridedStageFamily, StridedStageMemory, StridedTilingLayout},
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
/// Loads the content of all the stage using all planes,
/// keeping the original layout, making each tile strided
pub struct AsyncFullStridedLoading {}

impl LoadingValidation for AsyncFullStridedLoading {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        MatmulStridedLoading::check(client, problem, config, dtypes)
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullStridedLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: u8,
        plane_dim: u32,
        dtype: StorageType,
    ) -> u32 {
        MatmulStridedLoading::max_round_plane_count(
            elements_per_tile,
            tiles_per_stage,
            line_size,
            plane_dim,
            dtype,
        )
    }
}

#[cube]
impl FullLoadingStrategy for AsyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullStridedJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        runtime_args: RuntimeArgs,
        #[comptime] _line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let type_size = ES::type_size_bits();
        let line_size = comptime![ASYNC_COPY_WIDTH / type_size];
        let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
        let unit_count = config.loading_planes_count() * config.plane_dim;
        let num_tasks_per_unit = comptime!(num_stage_lines / unit_count);

        let unit_position_base = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;

        AsyncFullStridedJob {
            unit_position_base,
            runtime_args,
            num_tasks_per_unit,
            unit_count,
            copy_line_size: line_size,
        }
    }
}

#[derive(CubeType, Clone)]
pub struct AsyncFullStridedJob {
    unit_position_base: u32,
    runtime_args: RuntimeArgs,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    copy_line_size: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncCopy>
    for AsyncFullStridedJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, StridedTilingLayout>,
        _barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;
        let unit_position_abs = unit_position * this.copy_line_size;

        let layout = FullStageLayout::new(comptime![config.smem_config]);
        let view = global_iter.view();

        let pos = layout.to_source_pos(unit_position_abs);
        let stage_offset = unit_position_abs / stage.smem.line_size();

        async_copy_from(
            view,
            pos,
            stage,
            stage_offset,
            &this.runtime_args,
            global_iter.offset(),
            config,
            this.copy_line_size,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
