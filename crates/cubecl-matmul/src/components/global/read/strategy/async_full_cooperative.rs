use crate::components::{
    InvalidConfigError, MatmulIdent, MatrixLayout, MatrixPrecision, TilingScheme,
    global::{
        GlobalConfig,
        memory::{GlobalIterator, load_window_in_stage},
        multi_stage::LoadMaxRoundPlaneCount,
        read::{FullLoadingStrategy, LoadingJob, async_barrier::AsyncBarrier},
    },
    stage::{StridedStage, StridedStageFamily, StridedTilingLayout, TilingValidation},
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
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        StridedTilingLayout::check(config.global_memory_config(ident))?;

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
    type Job<IP: MatrixPrecision> = AsyncFullCooperativeJob;

    const SHOULD_CLEAR: bool = true;

    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] _line_size: u32,
        #[comptime] config: G,
    ) -> AsyncFullCooperativeJob {
        let matrix_layout = config.matrix_layout(ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.tiling_scheme().elements_in_stage_row(ident),
            MatrixLayout::ColMajor => config.tiling_scheme().elements_in_stage_col(ident),
        };

        AsyncFullCooperativeJob { num_slices, ident }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullCooperativeJob {
    #[cube(comptime)]
    num_slices: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<IP: MatrixPrecision> LoadingJob<IP, StridedTilingLayout, AsyncBarrier>
    for AsyncFullCooperativeJob
{
    type Stage = StridedStageFamily;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<IP::Global>>,
        stage: &mut StridedStage<IP::Stage, StridedTilingLayout>,
        barrier: &mut Barrier,
        #[comptime] config: G,
    ) {
        let window = load_window_in_stage(
            &global_iter.view(),
            task_id,
            comptime!(config.global_memory_config(this.ident)),
        );

        let mut destination: SliceMut<Line<IP::Stage>> = StridedTilingLayout::nth_slice::<IP::Stage>(
            stage,
            task_id,
            comptime!(config.stage_memory_config(this.ident)),
        );

        barrier.memcpy_async_cooperative(&window.try_cast_unchecked(), &mut destination);
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_slices
    }
}
