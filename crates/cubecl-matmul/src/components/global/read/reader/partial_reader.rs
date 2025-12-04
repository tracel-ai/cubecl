use super::StageBuffer;
use super::TaskCounter;
use crate::components::global::GlobalReaderConfig;
use crate::components::global::multi_stage::JobIterator;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::read::LoadingJob;
use crate::components::global::read::LoadingValidation;
use crate::components::global::read::SyncBarrier;
use crate::components::global::read::SyncStrategy;
use crate::components::stage::LoadStageFamily;
use crate::components::stage::StageFamily;
use crate::components::stage::TilingLayout;
use crate::components::{MatmulPrecision, global::memory::GlobalIterator};
use crate::components::{
    global::{SharedGlobalMatmulConfig, multi_stage::JobExecutor},
    stage::StageConfig,
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};

pub type LoaderStage<L, IP> = <<L as PartialLoadingStrategy>::Stage as StageFamily>::Stage<
    IP,
    <L as PartialLoadingStrategy>::TilingLayout,
>;

#[cube]
/// A strategy for loading partial stage memory
pub trait PartialLoadingStrategy:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;
    type SyncStrategy: SyncStrategy;
    type Stage: LoadStageFamily<ReadOnly>;

    /// The [LoadingJob] for this strategy.
    type Job<EG: Numeric, ES: Numeric>: LoadingJob<EG, ES, Self::TilingLayout, Self::SyncStrategy, Stage = Self::Stage>;

    /// Returns the job with preliminary calculations done.
    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES>;
}

#[cube]
/// A strategy for loading partial stage memory with async barriers. Used for specialized.
pub trait AsyncPartialLoadingStrategy:
    PartialLoadingStrategy<SyncStrategy: SyncStrategy<Barrier = Shared<Barrier>>>
{
    /// Arrival count for initializing the barrier
    fn arrival_count<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> u32;
    /// Extra synchronization after initializing the barrier, if needed
    fn barrier_post_init();
    /// Arrive at the barrier using the correct completion mechanism, without waiting
    fn arrive<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    );
    /// Whether this unit should participate in the load loop
    fn is_elected<S: StageConfig>(#[comptime] config: SharedGlobalMatmulConfig<S>) -> bool;
}

#[derive(Clone, CubeType)]
#[allow(clippy::type_complexity)]
/// Loads a stage from stage memory using synchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct PartialStageGlobalReader<EG: Numeric, ES: Numeric, L: PartialLoadingStrategy> {
    global_iter: GlobalIterator<Line<EG>>,
    stage_memory: LoaderStage<L, ES>,
    loading_job: CubeOption<(L::Job<EG, ES>, L::Job<EG, ES>)>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: PartialLoadingStrategy> PartialStageGlobalReader<EG, ES, L> {
    /// Create a new SyncPartialStageGlobalReader
    pub fn new(
        tensor: View<Line<EG>, Coords2d>,
        k_step: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self {
        let stage_memory = L::Stage::create(128u32, config.smem_config);
        let global_iter =
            GlobalIterator::new(tensor, k_step, config.gmem_config.view_direction, false);

        let loading_job = match config.precompute_job {
            true => CubeOption::new_Some((
                L::new_job::<EG, ES>(0u32, tensor.line_size(), config),
                L::new_job::<EG, ES>(1u32, tensor.line_size(), config),
            )),
            false => CubeOption::new_None(),
        };

        PartialStageGlobalReader::<EG, ES, L> {
            global_iter,
            stage_memory,
            loading_job,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(&self, #[comptime] stage_buffer: StageBuffer) -> LoaderStage<L, ES> {
        L::Stage::with_buffer_index(&self.stage_memory, stage_buffer.to_index())
    }

    /// Frees the stage memory for reuse
    pub fn free_stage(self) {
        L::Stage::free(&self.stage_memory);
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(
        &mut self,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut loading_job = match self.loading_job.clone() {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => L::new_job::<EG, ES>(0u32, self.global_iter.line_size(), config),
                StageBuffer::B => L::new_job::<EG, ES>(1u32, self.global_iter.line_size(), config),
            },
        };

        let len = L::Job::task_count(&loading_job);

        #[unroll]
        for task_id in 0..len {
            L::Job::<EG, ES>::execute_task(
                &mut loading_job,
                task_id,
                &self.global_iter,
                &mut self.stage_memory,
                barrier,
                config,
            );
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: PartialLoadingStrategy> JobExecutor<L::SyncStrategy>
    for PartialStageGlobalReader<EG, ES, L>
{
    type JobIterator = PartialJobIterator<EG, ES, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::JobIterator {
        let view = this.global_iter.view();
        let job = match this.loading_job.clone() {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => L::new_job::<EG, ES>(0u32, view.line_size(), config),
                StageBuffer::B => L::new_job::<EG, ES>(1u32, view.line_size(), config),
            },
        };

        let num_tasks = L::Job::task_count(&job);

        PartialJobIterator::<EG, ES, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut PartialJobIterator<EG, ES, L>,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let task_id = job_iterator.current.read().counter;

        L::Job::<EG, ES>::execute_task(
            &mut job_iterator.job,
            task_id,
            &this.global_iter,
            &mut this.stage_memory,
            barrier,
            config,
        );

        job_iterator.current.store(TaskCounter {
            counter: comptime!(task_id + 1u32),
        });
    }

    fn execute_all_remaining_tasks(
        this: &mut Self,
        job_iterator: &mut Self::JobIterator,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let task_counter = job_iterator.current.read().counter;

        #[unroll]
        for task_id in task_counter..job_iterator.num_tasks {
            L::Job::<EG, ES>::execute_task(
                &mut job_iterator.job,
                task_id,
                &this.global_iter,
                &mut this.stage_memory,
                barrier,
                config,
            );
        }

        job_iterator.current.store(TaskCounter {
            counter: comptime!(job_iterator.num_tasks),
        });
    }

    fn execute_whole_job(
        this: &mut Self,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) {
        Self::execute_all_remaining_tasks(
            this,
            &mut Self::create_job_iterator(this, stage_buffer, config),
            barrier,
            config,
        );
    }
}

#[derive(CubeType)]
/// Accomplish the entire job of filling the stage
pub struct PartialJobIterator<EG: Numeric, ES: Numeric, L: PartialLoadingStrategy> {
    job: L::Job<EG, ES>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: PartialLoadingStrategy> JobIterator
    for PartialJobIterator<EG, ES, L>
{
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
