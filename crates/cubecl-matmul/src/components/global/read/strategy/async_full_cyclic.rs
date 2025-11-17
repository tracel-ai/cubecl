use std::marker::PhantomData;

use crate::components::{
    InvalidConfigError, MatmulIdent, MatrixLayout, TilingScheme,
    global::{
        GlobalReaderConfig, RoleRule,
        memory::{GlobalIterator, load_window_in_tile},
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            FullLoadingStrategy, LoadingJob, async_barrier::AsyncBarrier, validate_async_barrier,
        },
    },
    stage::{ContiguousTilingLayout, StridedStage, TilingOrder, TilingValidation},
};
use cubecl_core::prelude::{barrier::Barrier, *};
use cubecl_core::{self as cubecl};

use super::LoadingValidation;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage memory using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct AsyncFullCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for AsyncFullCyclicLoading<T> {
    fn check<R: Runtime>(
        client: &ComputeClient<R::Server>,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        let total_units = config.loading_planes_count * config.plane_dim;
        let num_slices =
            config.smem_config.elements_in_tile_row * config.smem_config.tiles_in_stage();

        if num_slices >= total_units && !num_slices.is_multiple_of(total_units) {
            return Err(Box::new(format!(
                "Number of units ({total_units:?}) must divide number of slices ({num_slices:?}). Would require units doing different numbers of slices"
            )));
        }

        ContiguousTilingLayout::<T>::check(config.smem_config)?;
        validate_async_barrier::<R>(client)?;

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for AsyncFullCyclicLoading<TO> {
    fn max_round_plane_count(
        _tiling_scheme: &TilingScheme,
        _ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
    ) -> u32 {
        // Not sure what's ideal here, the current specialization isn't great anyways so can deal
        // with it later
        4
    }
}

#[cube]
impl<TO: TilingOrder> FullLoadingStrategy for AsyncFullCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = AsyncBarrier;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullCyclicJob;

    const SHOULD_CLEAR: bool = true;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncFullCyclicJob {
        let total_units = config.loading_units_count();

        let (num_slices_per_tile, slice_length_in_lines) = match config.gmem_config.matrix_layout {
            MatrixLayout::RowMajor => (
                config.smem_config.elements_in_tile_row,
                config.smem_config.elements_in_tile_col / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.smem_config.elements_in_tile_col,
                config.smem_config.elements_in_tile_row / line_size,
            ),
        };

        let num_slices = comptime!(num_slices_per_tile * config.smem_config.tiles_in_stage());
        let num_tasks_per_unit = num_slices.div_ceil(total_units);

        let unit_id = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;

        AsyncFullCyclicJob {
            unit_id,
            num_tasks_per_unit,
            total_units,
            num_slices,
            num_slices_per_tile,
            slice_length_in_lines,
            line_size,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullCyclicJob {
    unit_id: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    total_units: u32,
    #[cube(comptime)]
    num_slices: u32,
    #[cube(comptime)]
    num_slices_per_tile: u32,
    #[cube(comptime)]
    slice_length_in_lines: u32,
    #[cube(comptime)]
    line_size: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, AsyncBarrier> for AsyncFullCyclicJob
{
    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, ContiguousTilingLayout<TO>>,
        barrier: &mut Barrier,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let slice_index = this.unit_id + this.total_units * task_id;

        let nth_tile = slice_index / this.num_slices_per_tile;
        let (tile_x, tile_y) =
            ContiguousTilingLayout::<TO>::to_x_y(nth_tile, comptime!(config.smem_config));
        let nth_slice = slice_index % this.num_slices_per_tile;

        // TODO make branching comptime conditional (using Reader Mode)
        if slice_index < this.num_slices {
            let window = load_window_in_tile(
                &global_iter.view(),
                (tile_x, tile_y),
                nth_slice,
                config.smem_config,
                config.gmem_config,
            );

            // Where this unit writes source in the stage
            let slice_destination_offset =
                (nth_tile * this.num_slices_per_tile + nth_slice) * this.slice_length_in_lines;

            // Make destination start at offset
            let mut destination = stage.as_slice_mut(this.line_size).slice_mut(
                slice_destination_offset,
                slice_destination_offset + this.slice_length_in_lines,
            );

            barrier.memcpy_async(&window.try_cast_unchecked(), &mut destination);
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}
