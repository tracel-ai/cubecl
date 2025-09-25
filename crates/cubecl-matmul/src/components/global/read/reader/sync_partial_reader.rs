use std::marker::PhantomData;

use super::StageBuffer;
use super::TaskCounter;
use crate::components::InputPrecision;
use crate::components::MatmulIdent;
use crate::components::global::GlobalConfig;
use crate::components::global::memory::GlobalIterator;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::multi_stage::JobIterator;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::read::LoadingJob;
use crate::components::global::read::LoadingValidation;
use crate::components::stage::StridedStage;
use crate::components::stage::TilingLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};

#[cube]
/// A strategy for synchronously loading partial stage memory
pub trait SyncPartialLoadingStrategy:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<IP: InputPrecision>: LoadingJob<IP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<IP: InputPrecision, G: GlobalConfig>(
        #[comptime] stage_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<IP>;
}

#[derive(Clone, CubeType)]
/// Loads a stage from stage memory using synchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct SyncPartialStageGlobalReader<
    IP: InputPrecision,
    G: GlobalConfig,
    L: SyncPartialLoadingStrategy,
> {
    global_iter: GlobalIterator<IP::Global>,
    stage_memory: StridedStage<IP::Stage, L::TilingLayout>,
    loading_job: CubeOption<(L::Job<IP>, L::Job<IP>)>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<IP: InputPrecision, G: GlobalConfig, L: SyncPartialLoadingStrategy>
    SyncPartialStageGlobalReader<IP, G, L>
{
    /// Create a new SyncPartialStageGlobalReader
    pub fn new(
        tensor: View<Line<IP::Global>, Coords2d>,
        k_step: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self {
        let stage_memory = StridedStage::new(
            comptime!(ident.into_stage()),
            config.stage_memory_config(ident),
        );
        let global_iter = GlobalIterator::new(tensor, k_step, ident.view_direction(), false);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some((
                L::new_job::<IP, G>(0u32, ident, config),
                L::new_job::<IP, G>(1u32, ident, config),
            )),
            false => CubeOption::new_None(),
        };

        SyncPartialStageGlobalReader::<IP, G, L> {
            global_iter,
            stage_memory,
            loading_job,
            ident,
            _config: PhantomData::<G>,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(
        &self,
        #[comptime] stage_buffer: StageBuffer,
    ) -> StridedStage<IP::Stage, L::TilingLayout> {
        self.stage_memory.with_buffer_index(stage_buffer.to_index())
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(&mut self, #[comptime] stage_buffer: StageBuffer, #[comptime] config: G) {
        let mut loading_job = match self.loading_job {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => L::new_job::<IP, G>(0u32, self.ident, config),
                StageBuffer::B => L::new_job::<IP, G>(1u32, self.ident, config),
            },
        };

        let len = L::Job::task_count(&loading_job);

        let mut task_id = comptime![0u32];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..len {
            L::Job::<IP>::execute_task::<G>(
                &mut loading_job,
                task_id,
                &self.global_iter,
                &mut self.stage_memory,
                config,
            );
            comptime![task_id += 1];
        }
    }
}

#[cube]
impl<IP: InputPrecision, G: GlobalConfig, L: SyncPartialLoadingStrategy> JobExecutor<G>
    for SyncPartialStageGlobalReader<IP, G, L>
{
    type JobIterator = SyncPartialJobIterator<IP, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: G,
    ) -> Self::JobIterator {
        let job = match this.loading_job {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => L::new_job::<IP, G>(0u32, this.ident, config),
                StageBuffer::B => L::new_job::<IP, G>(1u32, this.ident, config),
            },
        };

        let num_tasks = L::Job::task_count(&job);

        SyncPartialJobIterator::<IP, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut SyncPartialJobIterator<IP, L>,
        #[comptime] config: G,
    ) {
        let task_id = job_iterator.current.read().counter;

        L::Job::<IP>::execute_task::<G>(
            &mut job_iterator.job,
            task_id,
            &this.global_iter,
            &mut this.stage_memory,
            config,
        );

        job_iterator.current.store(TaskCounter {
            counter: comptime!(task_id + 1u32),
        });
    }

    fn execute_all_remaining_tasks(
        this: &mut Self,
        job_iterator: &mut Self::JobIterator,
        #[comptime] config: G,
    ) {
        let task_counter = job_iterator.current.read().counter;

        let mut task_id = comptime![task_counter];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in task_counter..job_iterator.num_tasks {
            L::Job::<IP>::execute_task::<G>(
                &mut job_iterator.job,
                task_id,
                &this.global_iter,
                &mut this.stage_memory,
                config,
            );
            comptime![task_id += 1];
        }

        job_iterator.current.store(TaskCounter {
            counter: comptime!(job_iterator.num_tasks),
        });
    }

    fn execute_whole_job(
        this: &mut Self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: G,
    ) {
        Self::execute_all_remaining_tasks(
            this,
            &mut Self::create_job_iterator(this, stage_buffer, config),
            config,
        );
    }
}

#[derive(CubeType)]
/// Accomplish the entire job of filling the stage
pub struct SyncPartialJobIterator<IP: InputPrecision, L: SyncPartialLoadingStrategy> {
    job: L::Job<IP>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[cube]
impl<IP: InputPrecision, L: SyncPartialLoadingStrategy> JobIterator
    for SyncPartialJobIterator<IP, L>
{
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
