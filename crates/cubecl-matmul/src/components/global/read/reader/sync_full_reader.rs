use std::marker::PhantomData;

use crate::components::MatmulIdent;
use crate::components::MatrixPrecision;
use crate::components::global::GlobalConfig;
use crate::components::global::memory::GlobalIterator;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::multi_stage::JobIterator;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::read::LoadingJob;
use crate::components::global::read::LoadingValidation;
use crate::components::global::read::StageBuffer;
use crate::components::global::read::TaskCounter;
use crate::components::stage::StridedStage;
use crate::components::stage::TilingLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};

#[cube]
/// A strategy for synchronously loading a full stage memory.
pub trait SyncFullLoadingStrategy:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<IP: MatrixPrecision>: LoadingJob<IP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<IP>;
}

#[derive(Clone, CubeType)]
/// Loads the entire stage memory using synchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct SyncFullStageGlobalReader<
    IP: MatrixPrecision,
    G: GlobalConfig,
    L: SyncFullLoadingStrategy,
> {
    global_iter: GlobalIterator<Line<IP::Global>>,
    stage: StridedStage<IP::Stage, L::TilingLayout>,
    loading_job: CubeOption<L::Job<IP>>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(G, L)>,
}

#[cube]
impl<IP: MatrixPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy>
    SyncFullStageGlobalReader<IP, G, L>
{
    /// Create a new SyncFullStageGlobalReader
    pub fn new(
        tensor: View<Line<IP::Global>, Coords2d>,
        k_step: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self {
        let stage = StridedStage::new(
            comptime!(ident.into_stage()),
            config.stage_memory_config(ident),
        );
        let global_iter = GlobalIterator::new(tensor, k_step, ident.view_direction(), false);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<IP, G>(ident, config)),
            false => CubeOption::new_None(),
        };

        SyncFullStageGlobalReader::<IP, G, L> {
            global_iter,
            stage,
            loading_job,
            ident,
            _phantom: PhantomData::<(G, L)>,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(&self) -> StridedStage<IP::Stage, L::TilingLayout> {
        self.stage
    }

    pub fn free_stage(self) {
        unsafe { self.stage.free() };
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(&mut self, #[comptime] config: G) {
        let mut loading_job = match self.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<IP, G>(self.ident, config),
        };

        let len = L::Job::task_count(&loading_job);

        #[unroll]
        for task_id in 0..len {
            L::Job::<IP>::execute_task::<G>(
                &mut loading_job,
                task_id,
                &self.global_iter,
                &mut self.stage,
                config,
            );
        }
    }
}

#[cube]
impl<IP: MatrixPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> JobExecutor<G>
    for SyncFullStageGlobalReader<IP, G, L>
{
    type JobIterator = SyncFullStageJobIterator<IP, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] _stage_buffer: StageBuffer,
        #[comptime] config: G,
    ) -> Self::JobIterator {
        let job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<IP, G>(this.ident, config),
        };

        let num_tasks = L::Job::task_count(&job);

        SyncFullStageJobIterator::<IP, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut SyncFullStageJobIterator<IP, L>,
        #[comptime] config: G,
    ) {
        let task_id = job_iterator.current.read().counter;

        L::Job::<IP>::execute_task::<G>(
            &mut job_iterator.job,
            task_id,
            &this.global_iter,
            &mut this.stage,
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
                &mut this.stage,
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
/// A comptime iterator over a job for sync full stage reader
pub struct SyncFullStageJobIterator<IP: MatrixPrecision, L: SyncFullLoadingStrategy> {
    job: L::Job<IP>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[cube]
impl<IP: MatrixPrecision, L: SyncFullLoadingStrategy> JobIterator
    for SyncFullStageJobIterator<IP, L>
{
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
