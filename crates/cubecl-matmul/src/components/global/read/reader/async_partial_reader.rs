use super::StageBuffer;
use crate::components::global::CopyMechanism;
use crate::components::global::base::GlobalConfig;
use crate::components::global::memory::GlobalIterator;
use crate::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::components::global::read::{AsyncLoadingJob, LoadingValidation};
use crate::components::stage::TilingLayout;
use crate::components::stage::{self, StridedStage};
use crate::components::{MatrixPrecision, MatmulIdent};
use core::marker::PhantomData;
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};

#[cube]
/// A strategy for asynchronously loading a stage of stage memory
pub trait AsyncPartialLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<IP: MatrixPrecision>: AsyncLoadingJob<IP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] buffer_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<IP>;

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
/// Loads a stage from stage memory using asynchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct AsyncBufferGlobalReader<
    IP: MatrixPrecision,
    S: stage::StageConfig,
    CM: CopyMechanism,
    L: AsyncPartialLoadingStrategy,
> {
    tensor_reader: GlobalIterator<IP::Global>,
    stage_memory: StridedStage<IP::Stage, L::TilingLayout>,
    loading_job: CubeOption<(L::Job<IP>, L::Job<IP>)>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, CM)>,
}

#[cube]
impl<IP: MatrixPrecision, S: stage::StageConfig, CM: CopyMechanism, L: AsyncPartialLoadingStrategy>
    AsyncBufferGlobalReader<IP, S, CM, L>
{
    /// Create a new AsyncBufferGlobalReader
    pub fn new(
        tensor: View<Line<IP::Global>, Coords2d>,
        k_step: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) -> Self {
        let stage_memory = StridedStage::new(
            comptime!(ident.into_stage()),
            config.stage_memory_config(ident),
        );
        let tensor_reader = GlobalIterator::new(tensor, k_step, ident.view_direction(), true);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some((
                L::new_job::<IP, DoubleBufferingGlobalConfig<S>>(0u32, ident, config),
                L::new_job::<IP, DoubleBufferingGlobalConfig<S>>(1u32, ident, config),
            )),
            false => CubeOption::new_None(),
        };

        AsyncBufferGlobalReader::<IP, S, CM, L> {
            tensor_reader,
            stage_memory,
            loading_job,
            ident,
            _phantom: PhantomData::<(S, CM)>,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(
        &mut self,
        #[comptime] stage_buffer: StageBuffer,
    ) -> StridedStage<IP::Stage, L::TilingLayout> {
        self.stage_memory.with_buffer_index(stage_buffer.to_index())
    }

    /// Advance the view over global memory along the k dimension.
    pub fn advance_view(&mut self) {
        self.tensor_reader.advance();
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(
        &mut self,
        mechanism: &CM,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) {
        let mut loading_job = match self.loading_job {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => {
                    L::new_job::<IP, DoubleBufferingGlobalConfig<S>>(0u32, self.ident, config)
                }
                StageBuffer::B => {
                    L::new_job::<IP, DoubleBufferingGlobalConfig<S>>(1u32, self.ident, config)
                }
            },
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<IP>::execute_task::<CM, DoubleBufferingGlobalConfig<S>>(
                &mut loading_job,
                task_id,
                &self.tensor_reader,
                &mut self.stage_memory,
                mechanism,
                config,
            );
        }
    }

    /// Zero out the stage memory
    pub fn clear_stage(
        &mut self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) {
        self.stage_memory
            .clear_stage::<DoubleBufferingGlobalConfig<S>>(stage_buffer, self.ident, config)
    }
}
