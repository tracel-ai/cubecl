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
/// Loads global memory into the stage without layout change,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// Each `memcpy_async` is called with the same arguments for cooperative behaviour
pub struct AsyncFullCooperativeLoading {}

impl LoadingValidation for AsyncFullCooperativeLoading {
    fn check<R: Runtime>(
        client: &ComputeClient<R::Server>,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        StridedTilingLayout::check(config.global_memory_config)?;
        validate_async_barrier::<R>(client)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullCooperativeLoading {
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
impl FullLoadingStrategy for AsyncFullCooperativeLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncBarrier;
    type Job<EG: Numeric, ES: Numeric> = AsyncFullCooperativeJob;

    const SHOULD_CLEAR: bool = true;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] _line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> AsyncFullCooperativeJob {
        let matrix_layout = config.global_memory_config.matrix_layout;

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.stage_memory_config.elements_in_stage_row(),
            MatrixLayout::ColMajor => config.stage_memory_config.elements_in_stage_col(),
        };

        AsyncFullCooperativeJob { num_slices }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullCooperativeJob {
    #[cube(comptime)]
    num_slices: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LoadingJob<EG, ES, StridedTilingLayout, AsyncBarrier>
    for AsyncFullCooperativeJob
{
    fn execute_task(
        _this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, StridedTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let window = load_window_in_stage(
            &global_iter.view(),
            task_id,
            comptime!(config.global_memory_config),
        );

        let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES>(
            stage,
            task_id,
            comptime!(config.stage_memory_config),
        );

        barrier.memcpy_async_cooperative(&window.try_cast_unchecked(), &mut destination);
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_slices
    }
}
