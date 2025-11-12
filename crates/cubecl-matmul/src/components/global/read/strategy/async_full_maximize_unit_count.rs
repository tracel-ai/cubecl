use crate::components::{
    InvalidConfigError, MatmulIdent, MatrixLayout, TilingScheme,
    global::{
        GlobalReaderConfig,
        memory::{GlobalIterator, load_window_in_stage},
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            FullLoadingStrategy, LoadingJob, async_barrier::AsyncBarrier, validate_async_barrier,
        },
    },
    stage::{StridedStage, StridedTilingLayout, TilingValidation},
};
use cubecl_core::prelude::{barrier::Barrier, *};
use cubecl_core::{self as cubecl};

use super::LoadingValidation;

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct AsyncFullMaximizeUnitCountLoading {}

impl LoadingValidation for AsyncFullMaximizeUnitCountLoading {
    fn check<C: GlobalReaderConfig, R: Runtime>(
        client: &ComputeClient<R::Server>,
        config: &C,
        ident: MatmulIdent,
    ) -> Result<(), InvalidConfigError> {
        let matrix_layout = config.matrix_layout(ident);
        let line_size = config.global_line_size(ident);

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                config.tiling_scheme().elements_in_stage_row(ident),
                config.tiling_scheme().elements_in_stage_col(ident) / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.tiling_scheme().elements_in_stage_col(ident),
                config.tiling_scheme().elements_in_stage_row(ident) / line_size,
            ),
        };
        let unit_count = config.plane_dim() * config.num_loading_planes(ident);

        if !unit_count.is_multiple_of(num_slices) {
            return Err(Box::new(
                "Number of slices must divide number of units evenly",
            ));
        }
        if slice_length % (unit_count / num_slices) != 0 {
            return Err(Box::new(
                "Number of units per slice must divide slice length evenly",
            ));
        }

        StridedTilingLayout::check(config.global_memory_config(ident))?;
        validate_async_barrier::<R>(client)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullMaximizeUnitCountLoading {
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
impl FullLoadingStrategy for AsyncFullMaximizeUnitCountLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncBarrier;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullMaximizeUnitCountJob;

    const SHOULD_CLEAR: bool = true;

    fn new_job<EG: Numeric, ES: Numeric, G: GlobalReaderConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] line_size: u32,
        #[comptime] config: G,
    ) -> AsyncFullMaximizeUnitCountJob {
        let matrix_layout = config.matrix_layout(ident);

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                config.tiling_scheme().elements_in_stage_row(ident),
                config.tiling_scheme().elements_in_stage_col(ident) / line_size,
            ),
            MatrixLayout::ColMajor => (
                config.tiling_scheme().elements_in_stage_col(ident),
                config.tiling_scheme().elements_in_stage_row(ident) / line_size,
            ),
        };

        let unit_count = config.plane_dim() * config.num_loading_planes(ident);

        let units_per_slice = comptime!(unit_count / num_slices);
        let nth_slice = UNIT_POS / units_per_slice;

        let segment_length = comptime!(slice_length / units_per_slice);
        let nth_segment = UNIT_POS % units_per_slice;

        AsyncFullMaximizeUnitCountJob {
            nth_slice,
            nth_segment,
            segment_length,
            ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullMaximizeUnitCountJob {
    nth_slice: u32,
    nth_segment: u32,
    #[cube(comptime)]
    segment_length: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncBarrier>
    for AsyncFullMaximizeUnitCountJob
{
    fn execute_task<G: GlobalReaderConfig>(
        this: &mut Self,
        #[comptime] _task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, StridedTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: G,
    ) {
        let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES>(
            stage,
            this.nth_slice,
            comptime!(config.stage_memory_config(this.ident)),
        );

        let window = load_window_in_stage(
            &global_iter.view(),
            this.nth_slice,
            comptime!(config.global_memory_config(this.ident)),
        );
        let seg_start = Min::min(this.nth_segment * this.segment_length, window.len());
        let seg_end = Min::min((this.nth_segment + 1) * this.segment_length, window.len());

        let src_segment = window.slice(seg_start, seg_end);
        let mut dest_segment = destination.slice_mut(seg_start, seg_end);

        barrier.memcpy_async(&src_segment.try_cast_unchecked(), &mut dest_segment);
    }

    fn task_count(_this: &Self) -> comptime_type!(u32) {
        1
    }
}
