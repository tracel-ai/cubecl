use super::StageBuffer;
use crate::components::global::CopyMechanism;
use crate::components::global::base::GlobalConfig;
use crate::components::global::memory::GlobalIterator;
use crate::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::components::global::read::{AsyncLoadingJob, LoadingValidation};
use crate::components::stage::PartialStageReader;
use crate::components::stage::TilingLayout;
use crate::components::stage::{self, StageMemory};
use crate::components::{InputPrecision, MatmulIdent, StageIdent};
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
    type Job<IP: InputPrecision>: AsyncLoadingJob<IP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<IP: InputPrecision, G: GlobalConfig>(
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
    IP: InputPrecision,
    S: stage::StageConfig,
    CM: CopyMechanism,
    L: AsyncPartialLoadingStrategy,
> {
    tensor_reader: GlobalIterator<IP::Global>,
    stage_memory: StageMemory<IP::Stage, L::TilingLayout>,
    loading_job: CubeOption<(L::Job<IP>, L::Job<IP>)>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, CM)>,
}

#[cube]
impl<IP: InputPrecision, S: stage::StageConfig, CM: CopyMechanism, L: AsyncPartialLoadingStrategy>
    AsyncBufferGlobalReader<IP, S, CM, L>
{
    /// Create a new AsyncBufferGlobalReader
    pub fn new(
        tensor: View<Line<IP::Global>, Coords2d>,
        k_step: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) -> Self {
        let stage_memory = StageMemory::new::<S::StageMemoryConfig>(
            2u32,
            comptime!(ident.into_stage()),
            config.stage_memory_config(),
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
    pub fn stage_reader(
        this: &Self,
        #[comptime] stage_buffer: StageBuffer,
    ) -> PartialStageReader<IP::Stage, L::TilingLayout> {
        PartialStageReader::new(
            this.stage_memory,
            stage_buffer,
            comptime! {
                let stage_ident: StageIdent = this.ident.into();
                stage_ident
            },
        )
    }

    /// Advance the view over global memory along the k dimension.
    pub fn advance_view(this: &mut Self) {
        this.tensor_reader.advance();
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => {
                    L::new_job::<IP, DoubleBufferingGlobalConfig<S>>(0u32, this.ident, config)
                }
                StageBuffer::B => {
                    L::new_job::<IP, DoubleBufferingGlobalConfig<S>>(1u32, this.ident, config)
                }
            },
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<IP>::execute_task::<CM, DoubleBufferingGlobalConfig<S>>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                mechanism,
                config,
            );
        }
    }

    /// Zero out the stage memory
    pub fn clear_stage(
        this: &mut Self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) {
        this.stage_memory
            .clear_stage::<DoubleBufferingGlobalConfig<S>>(stage_buffer, this.ident, config)
    }
}
