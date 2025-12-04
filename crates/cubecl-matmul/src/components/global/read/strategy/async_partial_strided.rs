use crate::components::{InvalidConfigError, StageIdent};
use crate::components::{MatmulElems, global::read::validate_async_barrier};
use crate::components::{
    MatmulPrecision,
    global::{GlobalReaderConfig, RoleRule},
};
use crate::components::{
    MatmulProblem,
    global::{
        SharedGlobalMatmulConfig,
        read::{AsyncPartialLoadingStrategy, PartialLoadingStrategy, async_copy::ASYNC_COPY_WIDTH},
    },
};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::components::{global::read::async_copy::async_copy_from, stage::StridedStageMemory};
use crate::components::{global::read::stage::FullStageLayout, stage::StridedStageFamily};
use crate::components::{global::read::validate_swizzle_atom_size, stage::StageConfig};
use crate::components::{
    global::{
        multi_stage::LoadMaxRoundPlaneCount,
        read::{async_barrier::AsyncCopy, validate_async_copy},
    },
    stage::StridedTilingLayout,
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::tensor::layout::{Layout, LayoutExpand};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct AsyncPartialStridedLoading {}

impl LoadingValidation for AsyncPartialStridedLoading {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        let line_size =
            ASYNC_COPY_WIDTH / dtypes.stage(config.stage_ident.into()).size_bits() as u32;

        // Needs separate check because copy size may be larger than global line size
        if !config
            .smem_config
            .elements_per_stage_along_contiguous_dim()
            .is_multiple_of(line_size)
        {
            return Err(Box::new("Stage size isn't divisible by copy line size"));
        }

        let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
        let total_units = config.loading_units_count();

        if !num_stage_lines.is_multiple_of(total_units) {
            return Err(Box::new(format!(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {total_units:?} divides number of lines in stage.",
            )));
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        validate_async_barrier(client)?;
        validate_async_copy(client, problem, dtypes, config)?;
        StridedTilingLayout::check(config.smem_config)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialStridedLoading {
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
impl PartialLoadingStrategy for AsyncPartialStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, ES: Numeric> = AsyncPartialStridedJob;
    type Stage = StridedStageFamily;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
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

        AsyncPartialStridedJob {
            stage_index,
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            copy_line_size: line_size,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialStridedJob {
    unit_position_base: u32,

    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    copy_line_size: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncCopy>
    for AsyncPartialStridedJob
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
        let mut stage = stage.with_buffer_index(this.stage_index);

        let unit_position = this.unit_position_base + task_id * this.unit_count;
        let unit_position_abs = unit_position * this.copy_line_size;

        let layout = FullStageLayout::new(comptime![config.smem_config]);
        let view = global_iter.view();

        let pos = layout.to_source_pos(unit_position_abs);
        let pos = match config.stage_ident {
            StageIdent::Lhs => (
                pos.0,
                pos.1 + config.smem_config.elements_per_stage_along_col() * this.stage_index,
            ),
            StageIdent::Rhs => (
                pos.0 + config.smem_config.elements_per_stage_along_row() * this.stage_index,
                pos.1,
            ),
            _ => pos,
        };

        let stage_offset = unit_position_abs / stage.smem.line_size();

        async_copy_from(
            view,
            pos,
            &mut stage,
            stage_offset,
            config,
            this.copy_line_size,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
impl AsyncPartialLoadingStrategy for AsyncPartialStridedLoading {
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
