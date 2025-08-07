use crate::components::{
    InvalidConfigError, MatmulIdent, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig,
        load::AsyncFullLoadingStrategy,
        memory::{TensorReader, Window},
    },
    stage::{StageMemory, StridedTilingLayout},
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
    fn check<C: GlobalConfig>(_config: &C, _ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncFullLoadingStrategy for AsyncFullCooperativeLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision> = AsyncFullCooperativeJob;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
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
impl<MP: MatmulPrecision> AsyncLoadingJob<MP, StridedTilingLayout> for AsyncFullCooperativeJob {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, StridedTilingLayout>,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        let window: Window<MP::EI> = tensor_reader
            .load_window_in_stage(task_id, comptime!(config.global_memory_config(this.ident)));
        let mut destination: SliceMut<Line<MP::ES>> =
            StridedTilingLayout::nth_slice::<MP::ES, G::StageMemoryConfig>(
                stage,
                task_id,
                comptime!(this.ident.into_stage()),
                config.stage_memory_config(),
            );

        CM::memcpy_async(
            mechanism,
            &window.slice.try_cast_unchecked(),
            &mut destination,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_slices
    }
}
