use std::marker::PhantomData;

use crate::components::StageIdent;
use crate::components::global::GlobalReaderConfig;
use crate::components::global::memory::GlobalIterator;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::multi_stage::JobIterator;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::read::LoadingJob;
use crate::components::global::read::LoadingValidation;
use crate::components::global::read::StageBuffer;
use crate::components::global::read::SyncStrategy;
use crate::components::global::read::TaskCounter;
use crate::components::stage::StridedStageFamily;
use crate::components::stage::StridedStageMemory;
use crate::components::stage::TilingLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};

pub type SyncBarrier<S> = <S as SyncStrategy>::Barrier;

#[cube]
/// A strategy for synchronously loading a full stage memory.
pub trait FullLoadingStrategy:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;
    /// The synchronization strategy that should be used with this loading strategy
    type SyncStrategy: SyncStrategy;

    /// The [LoadingJob] for this strategy.
    type Job<EG: Numeric, ES: Numeric>: LoadingJob<EG, ES, Self::TilingLayout, Self::SyncStrategy, Stage = StridedStageFamily>;

    const SHOULD_CLEAR: bool = false;

    /// Returns the job with preliminary calculations done.
    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES>;
}

#[derive(Clone, CubeType)]
/// Loads the entire stage memory.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct FullStageGlobalReader<EG: Numeric, ES: Numeric, L: FullLoadingStrategy> {
    global_iter: GlobalIterator<Line<EG>>,
    stage: StridedStageMemory<ES, L::TilingLayout>,
    loading_job: CubeOption<L::Job<EG, ES>>,
    #[cube(comptime)]
    _phantom: PhantomData<L>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: FullLoadingStrategy> FullStageGlobalReader<EG, ES, L> {
    /// Create a new SyncFullStageGlobalReader
    pub fn new(
        view: View<Line<EG>, Coords2d>,
        k_step: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self {
        // Maybe make align a property on the strategy, but it's fine to over-align so this works
        // for now. Swizzling will require more though.
        let mut stage = StridedStageMemory::new_aligned(128u32, config.smem_config);

        let (shape_row, shape_col) = view.shape();
        let global_iter =
            GlobalIterator::new(view, k_step, config.gmem_config.view_direction, false);

        let loading_job = match config.precompute_job {
            true => CubeOption::new_Some(L::new_job::<EG, ES>(view.line_size(), config)),
            false => CubeOption::new_None(),
        };

        if L::SHOULD_CLEAR {
            // Slices are clamped to the shape, so if the slice size is smaller than the stage size
            // we are partially out of bounds.
            match config.stage_ident {
                StageIdent::Lhs =>
                {
                    #[allow(clippy::collapsible_if)]
                    if config.gmem_config.check_row_bounds {
                        if shape_row < config.smem_config.elements_per_stage_along_row() {
                            stage.clear_all(config);
                        }
                    }
                }
                StageIdent::Rhs =>
                {
                    #[allow(clippy::collapsible_if)]
                    if config.gmem_config.check_col_bounds {
                        if shape_col < config.smem_config.elements_per_stage_along_col() {
                            stage.clear_all(config);
                        }
                    }
                }
                _ => comptime!(unreachable!()),
            }
        }

        FullStageGlobalReader::<EG, ES, L> {
            global_iter,
            stage,
            loading_job,
            _phantom: PhantomData::<L>,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(&self) -> StridedStageMemory<ES, L::TilingLayout> {
        self.stage
    }

    pub fn clear_stage(&mut self, #[comptime] config: GlobalReaderConfig) {
        self.stage.clear_all(config);
    }

    pub fn free_stage(self) {
        unsafe { self.stage.free() };
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(
        &mut self,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut loading_job = match self.loading_job.clone() {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<EG, ES>(self.global_iter.line_size(), config),
        };

        let len = L::Job::task_count(&loading_job);

        #[unroll]
        for task_id in 0..len {
            L::Job::<EG, ES>::execute_task(
                &mut loading_job,
                task_id,
                &self.global_iter,
                &mut self.stage,
                barrier,
                config,
            );
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: FullLoadingStrategy> JobExecutor<L::SyncStrategy>
    for FullStageGlobalReader<EG, ES, L>
{
    type JobIterator = FullStageJobIterator<EG, ES, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] _stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::JobIterator {
        let view = this.global_iter.view();
        let job = match this.loading_job.clone() {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<EG, ES>(view.line_size(), config),
        };

        let num_tasks = L::Job::task_count(&job);

        FullStageJobIterator::<EG, ES, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut FullStageJobIterator<EG, ES, L>,
        barrier: &mut SyncBarrier<L::SyncStrategy>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let task_id = job_iterator.current.read().counter;

        L::Job::<EG, ES>::execute_task(
            &mut job_iterator.job,
            task_id,
            &this.global_iter,
            &mut this.stage,
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
                &mut this.stage,
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
/// A comptime iterator over a job for sync full stage reader
pub struct FullStageJobIterator<EG: Numeric, ES: Numeric, L: FullLoadingStrategy> {
    job: L::Job<EG, ES>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: FullLoadingStrategy> JobIterator
    for FullStageJobIterator<EG, ES, L>
{
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
