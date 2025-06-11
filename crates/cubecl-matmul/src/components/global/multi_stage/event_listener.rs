use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::InputIdent;
use crate::components::global::load::BufferId;
use crate::components::global::{GlobalConfig, LoadingSides};
use crate::components::stage::{StageEvent, StageEventListener};

#[derive(Copy, Clone)]
pub enum EventLoadingMode {
    // Load without constraints
    Relaxed,
    // Load but respecting order
    Ordered,
}

#[derive(CubeType)]
pub struct DoubleBufferingEventListener<Lhs: JobExecutor<G>, Rhs: JobExecutor<G>, G: GlobalConfig> {
    #[cube(comptime)]
    buffer_id: BufferId,
    loader_lhs: Lhs,
    loader_rhs: Rhs,
    #[cube(comptime)]
    config: G,
    state_lhs: Sequence<Lhs::Job>,
    state_rhs: Sequence<Rhs::Job>,
    #[cube(comptime)]
    event_loading_side: LoadingSides,
}

#[derive(Clone)]
struct IdentEventAnalysis {
    /// The event count to execute the next task.
    counter: u32,
    /// If tasks can start without risk of concurrency
    can_start: bool,
    /// If no more tasks need to be executed .
    completed: bool,
}
impl CubeDebug for IdentEventAnalysis {}
impl IdentEventAnalysis {
    fn should_execute(&self, current: u32) -> bool {
        self.counter == current && self.can_start && !self.completed
    }
}

#[derive(Clone)]
/// Analysis of [StageEvent] that reports when lhs and rhs should execute a task.
struct EventAnalysis {
    lhs: IdentEventAnalysis,
    rhs: IdentEventAnalysis,
}
impl CubeDebug for EventAnalysis {}

#[cube]
impl<Lhs: JobExecutor<G>, Rhs: JobExecutor<G>, G: GlobalConfig>
    DoubleBufferingEventListener<Lhs, Rhs, G>
{
    pub fn new(
        #[comptime] buffer_id: BufferId,
        loader_lhs: &Lhs,
        loader_rhs: &Rhs,
        #[comptime] config: G,
        #[comptime] event_loading_side: LoadingSides,
    ) -> DoubleBufferingEventListener<Lhs, Rhs, G> {
        DoubleBufferingEventListener::<Lhs, Rhs, G> {
            buffer_id,
            loader_lhs: comptime![loader_lhs.clone()],
            loader_rhs: comptime![loader_rhs.clone()],
            config,
            state_lhs: Sequence::new(),
            state_rhs: Sequence::new(),
            event_loading_side,
        }
    }
}

#[cube]
impl<L: JobExecutor<G>, R: JobExecutor<G>, G: GlobalConfig> StageEventListener
    for DoubleBufferingEventListener<L, R, G>
{
    fn on_event(this: &mut Self, #[comptime] event: StageEvent) {
        if let StageEvent::Begin = event {
            this.init();
        }

        if let StageEvent::TmmCompleted { current, total } = event {
            let analysis = this.analyse(current, total);

            if comptime![analysis.lhs.should_execute(current)] {
                let lhs_job = this.state_lhs.index_mut(0);

                // #[cfg(target_os = "macos")]
                // sync_plane();

                L::execute_task(&mut this.loader_lhs, lhs_job, this.config);
            }

            if comptime![analysis.rhs.should_execute(current)] {
                let rhs_job = this.state_rhs.index_mut(0);

                // #[cfg(target_os = "macos")]
                // sync_plane();

                R::execute_task(&mut this.loader_rhs, rhs_job, this.config);
            }
        }

        // Cleanup remaining tasks if any.
        if let StageEvent::Finish = event {
            let lhs_len = this.state_lhs.len();
            let rhs_len = this.state_rhs.len();

            let mut lhs_num_tasks = comptime!(0u32);
            let mut rhs_num_tasks = comptime!(0u32);
            let mut lhs_num_task_executed = comptime!(0u32);
            let mut rhs_num_task_executed = comptime!(0u32);

            if comptime!(lhs_len > 0) {
                let lhs_job = this.state_lhs.index_mut(0);
                let num_tasks = L::Job::num_tasks(lhs_job);
                let num_task_executed = L::Job::current(lhs_job);
                comptime!(lhs_num_tasks += num_tasks);
                comptime!(lhs_num_task_executed += num_task_executed);
            }

            if comptime!(rhs_len > 0) {
                let rhs_job = this.state_rhs.index_mut(0);
                let num_tasks = R::Job::num_tasks(rhs_job);
                let num_task_executed = R::Job::current(rhs_job);
                comptime!(rhs_num_tasks += num_tasks);
                comptime!(rhs_num_task_executed += num_task_executed);
            }

            // #[cfg(target_os = "macos")]
            // if lhs_num_tasks - lhs_num_task_executed + rhs_num_tasks - rhs_num_task_executed > 0 {
            //     sync_plane();
            // }

            if comptime!(lhs_len > 0) {
                let lhs_job = this.state_lhs.index_mut(0);
                #[unroll]
                for _ in lhs_num_task_executed..lhs_num_tasks {
                    L::execute_task(&mut this.loader_lhs, lhs_job, this.config);
                }
            }

            if comptime!(rhs_len > 0) {
                let rhs_job = this.state_rhs.index_mut(0);
                #[unroll]
                for _ in rhs_num_task_executed..rhs_num_tasks {
                    R::execute_task(&mut this.loader_rhs, rhs_job, this.config);
                }
            }
        }
    }
}

#[cube]
impl<L: JobExecutor<G>, R: JobExecutor<G>, G: GlobalConfig> DoubleBufferingEventListener<L, R, G> {
    fn init(&mut self) {
        if comptime!(self.event_loading_side.includes_lhs()) {
            self.state_lhs
                .push(L::create_job(&self.loader_lhs, self.buffer_id, self.config));
        }

        if comptime!(self.event_loading_side.includes_rhs()) {
            self.state_rhs
                .push(R::create_job(&self.loader_rhs, self.buffer_id, self.config));
        }
    }

    fn analyse(
        &self,
        #[comptime] current_event: u32,
        #[comptime] event_count_total: u32,
    ) -> comptime_type!(EventAnalysis) {
        let lhs_len = self.state_lhs.len();
        let rhs_len = self.state_rhs.len();

        let mut lhs_num_tasks = comptime!(0u32);
        let mut rhs_num_tasks = comptime!(0u32);
        let mut lhs_num_task_executed = comptime!(0u32);
        let mut rhs_num_task_executed = comptime!(0u32);

        if comptime!(lhs_len > 0) {
            let lhs_job = self.state_lhs.index(0);
            let num_tasks = L::Job::num_tasks(lhs_job);
            let current = L::Job::current(lhs_job);
            comptime!(lhs_num_tasks += num_tasks);
            comptime!(lhs_num_task_executed += current);
        }

        if comptime!(rhs_len > 0) {
            let rhs_job = self.state_rhs.index(0);
            let num_tasks = R::Job::num_tasks(rhs_job);
            let current = R::Job::current(rhs_job);
            comptime!(rhs_num_tasks += num_tasks);
            comptime!(rhs_num_task_executed += current);
        }

        let num_tasks_total = comptime!(lhs_num_tasks + rhs_num_tasks);

        comptime! {
            // When ordered, we cannot start loading before all were loaded in fragments
            // Eventually, Lhs loads for k = i could start as soon as k_iterations_done = i, but probably overkill
            let can_start = |event_loading_mode: EventLoadingMode| if let EventLoadingMode::Ordered = event_loading_mode {
                current_event >= event_count_total - self.config.tiling_scheme().tiles_in_stage_partition_mn()
            } else {
                true
            };

            let lhs_can_start = lhs_len > 0 && can_start(self.config.event_loading_mode(InputIdent::Lhs));
            let rhs_can_start = rhs_len > 0 && can_start(self.config.event_loading_mode(InputIdent::Rhs));

            let step = 1u32;
            let start = event_count_total.saturating_sub(step * num_tasks_total);

            EventAnalysis {
                lhs: IdentEventAnalysis{
                    counter: lhs_num_task_executed * step + start,
                    can_start: lhs_can_start,
                    completed: lhs_num_task_executed >= lhs_num_tasks,
                },
                rhs: IdentEventAnalysis {
                    counter: rhs_num_task_executed * step + (lhs_num_tasks * step) + start,
                    can_start: rhs_can_start,
                    completed: rhs_num_task_executed >= rhs_num_tasks,
                }
            }
        }
    }
}

#[cube]
pub trait JobExecutor<G: GlobalConfig>: CubeType + Clone {
    type Job: Job;

    fn create_job(this: &Self, #[comptime] buffer_id: BufferId, #[comptime] config: G)
    -> Self::Job;

    fn execute_task(this: &mut Self, job: &mut Self::Job, #[comptime] config: G);

    fn execute_all_remaining_tasks(this: &mut Self, job: &mut Self::Job, #[comptime] config: G);

    fn execute_whole_job(this: &mut Self, #[comptime] buffer_id: BufferId, #[comptime] config: G);
}

#[cube]
pub trait Job: CubeType {
    fn current(this: &Self) -> comptime_type!(u32);
    fn num_tasks(this: &Self) -> comptime_type!(u32);
}
