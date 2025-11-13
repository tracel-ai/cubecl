use std::marker::PhantomData;

use super::StageBuffer;
use super::TaskCounter;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::multi_stage::JobIterator;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::read::LoadingJob;
use crate::components::global::read::LoadingValidation;
use crate::components::global::read::SyncStrategy;
use crate::components::global::{GlobalConfig, read::SyncBarrier};
use crate::components::stage::TilingLayout;
use crate::components::{MatmulIdent, stage::StageFamily};
use crate::components::{global::memory::GlobalIterator, stage::LoadStageFamily};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
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
    fn new_job<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        #[comptime] stage_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] line_size: u32,
        #[comptime] config: G,
    ) -> Self::Job<EG, ES>;
}

#[derive(Clone, CubeType)]
#[allow(clippy::type_complexity)]
/// Loads a stage from stage memory using synchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct PartialStageGlobalReader<
    EG: Numeric,
    ES: Numeric,
    G: GlobalConfig,
    L: PartialLoadingStrategy,
> {
    global_iter: GlobalIterator<Line<EG>>,
    stage_memory: LoaderStage<L, ES>,
    loading_job: CubeOption<(L::Job<EG, ES>, L::Job<EG, ES>)>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: GlobalConfig, L: PartialLoadingStrategy>
    PartialStageGlobalReader<EG, ES, G, L>
{
    /// Create a new SyncPartialStageGlobalReader
    pub fn new(
        tensor: View<Line<EG>, Coords2d>,
        k_step: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self {
        let stage_memory = L::Stage::create(128u32, config.stage_memory_config(ident));
        let global_iter = GlobalIterator::new(tensor, k_step, ident.view_direction(), false);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some((
                L::new_job::<EG, ES, G>(0u32, ident, tensor.line_size(), config),
                L::new_job::<EG, ES, G>(1u32, ident, tensor.line_size(), config),
            )),
            false => CubeOption::new_None(),
        };

        PartialStageGlobalReader::<EG, ES, G, L> {
            global_iter,
            stage_memory,
            loading_job,
            ident,
            _config: PhantomData::<G>,
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
        #[comptime] config: G,
    ) {
        let mut loading_job = match self.loading_job {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => {
                    L::new_job::<EG, ES, G>(0u32, self.ident, self.global_iter.line_size(), config)
                }
                StageBuffer::B => {
                    L::new_job::<EG, ES, G>(1u32, self.ident, self.global_iter.line_size(), config)
                }
            },
        };

        let len = L::Job::task_count(&loading_job);

        #[unroll]
        for task_id in 0..len {
            L::Job::<EG, ES>::execute_task::<G>(
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
impl<EG: Numeric, ES: Numeric, G: GlobalConfig, L: PartialLoadingStrategy>
    JobExecutor<G, L::SyncStrategy> for PartialStageGlobalReader<EG, ES, G, L>
{
    type JobIterator = PartialJobIterator<EG, ES, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: G,
    ) -> Self::JobIterator {
        let view = this.global_iter.view();
        let job = match this.loading_job {
            CubeOption::Some(job) => match stage_buffer {
                StageBuffer::A => job.0,
                StageBuffer::B => job.1,
            },
            CubeOption::None => match stage_buffer {
                StageBuffer::A => {
                    L::new_job::<EG, ES, G>(0u32, this.ident, view.line_size(), config)
                }
                StageBuffer::B => {
                    L::new_job::<EG, ES, G>(1u32, this.ident, view.line_size(), config)
                }
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
        #[comptime] config: G,
    ) {
        let task_id = job_iterator.current.read().counter;

        L::Job::<EG, ES>::execute_task::<G>(
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
        #[comptime] config: G,
    ) {
        let task_counter = job_iterator.current.read().counter;

        #[unroll]
        for task_id in task_counter..job_iterator.num_tasks {
            L::Job::<EG, ES>::execute_task::<G>(
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
        #[comptime] config: G,
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
