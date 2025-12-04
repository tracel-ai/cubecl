use crate::components::{
    InvalidConfigError, MatmulElems, MatmulProblem, MatrixLayout,
    global::{
        GlobalReaderConfig,
        memory::{GlobalIterator, load_window_in_stage},
        multi_stage::LoadMaxRoundPlaneCount,
        read::{
            FullLoadingStrategy, LoadingJob, async_barrier::AsyncBarrier, validate_async_barrier,
            validate_noswizzle,
        },
    },
    stage::{StridedStageFamily, StridedStageMemory, StridedTilingLayout, TilingValidation},
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
        client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        _dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        StridedTilingLayout::check(config.smem_config)?;
        validate_async_barrier(client)?;
        validate_noswizzle(config.smem_config)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncFullCooperativeLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        _tiles_per_stage: u32,
        _line_size: u8,
        _plane_dim: u32,
        _dtype: StorageType,
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
        let matrix_layout = config.gmem_config.matrix_layout;

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.smem_config.elements_per_stage_along_row(),
            MatrixLayout::ColMajor => config.smem_config.elements_per_stage_along_col(),
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
    type Stage = StridedStageFamily;

    fn execute_task(
        _this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, StridedTilingLayout>,
        barrier: &mut Shared<Barrier>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let window = load_window_in_stage(
            &global_iter.view(),
            task_id,
            config.smem_config,
            config.gmem_config,
        );

        let mut destination: SliceMut<Line<ES>> =
            StridedTilingLayout::nth_slice::<ES>(stage, task_id, comptime!(config.smem_config));

        barrier.memcpy_async_cooperative(&window.try_cast_unchecked(), &mut destination);
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_slices
    }
}
