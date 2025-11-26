use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::read::{FullLoadingStrategy, stage::FullStageLayout};
use crate::components::global::{GlobalReaderConfig, RoleRule};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{StridedStageMemory, StridedTilingLayout};
use crate::components::{
    InvalidConfigError,
    global::read::{validate_async_barrier, validate_async_copy},
};
use crate::components::{MatmulElems, global::read::async_barrier::AsyncCopy};
use crate::components::{MatrixLayout, global::read::validate_swizzle_atom_size};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use cubecl_core::prelude::{
    barrier::{copy_async, copy_async_checked},
    *,
};
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::{
    tensor::layout::{Layout, LayoutExpand},
    type_size,
};

use super::{LoadingJob, LoadingValidation};

/// The instruction has a max width of 128 bits, even on Blackwell which supports 256-bit loads
const LOAD_WIDTH: u32 = 128;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the stage using all planes,
/// keeping the original layout, making each tile strided
pub struct AsyncFullStridedLoading {}

impl LoadingValidation for AsyncFullStridedLoading {
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        let line_size = LOAD_WIDTH / dtypes.stage(config.stage_ident.into()).size_bits() as u32;

        let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
        let total_units = config.loading_units_count();

        if !num_stage_lines.is_multiple_of(total_units) {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        validate_async_barrier(client)?;
        validate_async_copy(client, dtypes, config)?;
        StridedTilingLayout::check(config.smem_config)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullStridedLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        line_size: u8,
        plane_dim: u32,
    ) -> u32 {
        let elements_per_stage = elements_per_tile * tiles_per_stage;
        let num_lines = elements_per_stage / line_size as u32;
        num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl FullLoadingStrategy for AsyncFullStridedLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncCopy;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullStridedJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] _line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let type_size = ES::type_size_bits();
        let line_size = comptime![LOAD_WIDTH / type_size];

        let num_stage_lines = config.smem_config.elements_per_stage() / line_size;
        let unit_count = config.loading_planes_count() * config.plane_dim;
        let num_tasks_per_unit = comptime!(num_stage_lines / unit_count);

        let unit_position_base = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;

        AsyncFullStridedJob {
            unit_position_base,
            num_tasks_per_unit,
            unit_count,
            copy_line_size: line_size,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullStridedJob {
    unit_position_base: u32,

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
        _barrier: &mut Barrier,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let unit_position = this.unit_position_base + task_id * this.unit_count;

        let layout = FullStageLayout::new(comptime![config.smem_config]);
        let view = global_iter.view();

        let pos = layout.to_source_pos(unit_position * this.copy_line_size);

        let mut slice = stage.as_slice_mut(stage.smem.line_size());
        let mut slice_len_global = comptime![this.copy_line_size / view.line_size()].runtime();
        let slice_len_stage = this.copy_line_size / slice.line_size();

        if config.gmem_config.check_row_bounds {
            let pos = pos.0;
            let shape = view.shape().0;
            match config.gmem_config.matrix_layout {
                MatrixLayout::RowMajor => {
                    slice_len_global *= u32::cast_from(pos < shape);
                }
                MatrixLayout::ColMajor => {
                    slice_len_global =
                        Min::min(SaturatingSub::saturating_sub(shape, pos), slice_len_global);
                }
            }
        }

        if config.gmem_config.check_col_bounds {
            let pos = pos.1;
            let shape = view.shape().1;
            match config.gmem_config.matrix_layout {
                MatrixLayout::RowMajor => {
                    slice_len_global =
                        Min::min(SaturatingSub::saturating_sub(shape, pos), slice_len_global);
                }
                MatrixLayout::ColMajor => {
                    slice_len_global *= u32::cast_from(pos < shape);
                }
            }
        }

        let slice_slize = comptime![match config.smem_config.matrix_layout {
            MatrixLayout::RowMajor => (1u32, this.copy_line_size),
            MatrixLayout::ColMajor => (this.copy_line_size, 1u32),
        }]
        .runtime();

        let global_slice = view.slice_unchecked(pos, slice_slize).to_linear_slice();

        let type_size = type_size::<ES>(slice.line_size());
        let stage_offs = stage.swizzle.apply(unit_position, type_size);

        let stage_slice = slice.slice_mut(stage_offs, stage_offs + slice_len_stage);

        if comptime![config.gmem_config.check_row_bounds || config.gmem_config.check_col_bounds] {
            copy_async_checked(
                &global_slice.slice(0, slice_len_global),
                &mut stage_slice.try_cast_unchecked(),
                this.copy_line_size,
            );
        } else {
            copy_async(
                &global_slice,
                &mut stage_slice.try_cast_unchecked(),
                this.copy_line_size,
            );
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
