use crate::components::global::GlobalConfig;
use crate::components::global::load::Task;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct TaskQueue<T: Task> {
    seq: Sequence<T>,
    cursor: TaskCursor,
}

#[cube]
impl<T: Task> TaskQueue<T> {
    pub fn init(seq: Sequence<T>) -> TaskQueue<T> {
        TaskQueue::<T> {
            seq,
            cursor: TaskCursor::new(),
        }
    }

    /// Returns true if executed, false if was end of sequence
    pub fn execute_next<G: GlobalConfig>(
        &mut self,
        executor: &mut T::Executor<G>,
        #[comptime] config: G,
    ) {
        let current = self.cursor.current_value();
        if current < self.seq.len() {
            let mut task = self.seq.index_mut(current);
            <T as Task>::execute(&mut task, executor, config);

            self.cursor.increment();
        }
    }
}

#[derive(CubeType)]
pub struct AllTasksQueue<LT: Task, RT: Task> {
    lhs: TaskQueue<LT>,
    rhs: TaskQueue<RT>,
    cursor: TaskCursor,
}

impl<LT: Task, RT: Task> AllTasksQueue<LT, RT> {
    pub fn init(lhs: TaskQueue<LT>, rhs: TaskQueue<RT>) -> AllTasksQueue<LT, RT> {
        AllTasksQueue {
            lhs,
            rhs,
            cursor: TaskCursor::new(),
        }
    }

    pub fn execute_next<G: GlobalConfig>(
        &self,
        lhs_executor: LT::Executor<G>,
        rhs_executor: RT::Executor<G>,
    ) {
    }
}

#[derive(CubeType, Clone)]
pub struct TaskCursor {
    #[cube(comptime)]
    pub index: ComptimeCell<u32>,
}

#[cube]
impl TaskCursor {
    pub fn new() -> TaskCursor {
        TaskCursor {
            index: ComptimeCell::new(0),
        }
    }

    pub fn increment(&mut self) {
        self.index.store(self.index.read() + 1);
    }

    pub fn current_value(&self) -> u32 {
        self.index.read()
    }
}

#[derive(Clone)]
/// Determines what is loaded interleaved with computation
pub enum QueueConfig {
    /// Loads Lhs and Rhs buffers
    Full,
    /// Loads Lhs buffer only
    LhsOnly,
    /// Loads Rhs buffer only
    RhsOnly,
    /// Doesn't load anything
    /// fill_stage will need to be called instead
    None,
}

impl QueueConfig {
    pub fn should_fill_lhs(&self) -> bool {
        match self {
            QueueConfig::Full => true,
            QueueConfig::LhsOnly => true,
            QueueConfig::RhsOnly => false,
            QueueConfig::None => false,
        }
    }

    pub fn should_fill_rhs(&self) -> bool {
        match self {
            QueueConfig::Full => true,
            QueueConfig::LhsOnly => false,
            QueueConfig::RhsOnly => true,
            QueueConfig::None => false,
        }
    }
}
