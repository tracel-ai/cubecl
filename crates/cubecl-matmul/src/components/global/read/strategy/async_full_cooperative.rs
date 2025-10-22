use crate::components::{
    InvalidConfigError, MatmulIdent, MatrixLayout, MatrixPrecision,
    global::{
        CopyMechanism, GlobalConfig,
        memory::{GlobalIterator, load_window_in_stage},
        read::AsyncFullLoadingStrategy,
    },
    stage::{StridedStage, StridedTilingLayout, TilingValidation},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::{AsyncLoadingJob, LoadingValidation};

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

#[cube]
impl AsyncFullLoadingStrategy for AsyncFullCooperativeLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<IP: MatrixPrecision> = AsyncFullCooperativeJob;

    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> AsyncFullCooperativeJob {
        let matrix_layout = config.matrix_layout(ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => config.tiling_scheme().elements_in_stage_row(ident),
            MatrixLayout::ColMajor => config.tiling_scheme().elements_in_stage_col(ident),
        };

        AsyncFullCooperativeJob { num_slices, ident }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
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
impl<IP: MatrixPrecision> AsyncLoadingJob<IP, StridedTilingLayout> for AsyncFullCooperativeJob {
    fn execute_task<CM: CopyMechanism, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        global_iter: &GlobalIterator<Line<IP::Global>>,
        stage: &mut StridedStage<IP::Stage, StridedTilingLayout>,
        mechanism: &CM,
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

        CM::memcpy_async(mechanism, &window.try_cast_unchecked(), &mut destination);
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_slices
    }
}
