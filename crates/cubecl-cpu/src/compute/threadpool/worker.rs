use std::sync::Arc;
use std::thread;

use crate::compute::{
    affinity::{CoreId, set_for_current},
    threadpool::compute_task::ComputeTask,
    threadpool::thread_buffer::ThreadBuffer,
};

pub const MAX_STACK_SIZE: usize = 16 * 1024 * 1024;
pub const DEFAULT_STACK_SIZE: usize = 64 * 1024 * 1024;

fn resolve_stack_size() -> usize {
    if let Ok(value) = std::env::var("CUBECL_CPU_STACK_SIZE")
        && let Ok(bytes) = value.parse::<usize>()
    {
        return bytes.max(MAX_STACK_SIZE);
    }
    if let Ok(value) = std::env::var("CUBECL_CPU_STACK_MB")
        && let Ok(mb) = value.parse::<usize>()
    {
        return (mb.saturating_mul(1024 * 1024)).max(MAX_STACK_SIZE);
    }
    DEFAULT_STACK_SIZE
}

pub struct Worker {
    threads_buffer: Arc<[spin::Mutex<ThreadBuffer<ComputeTask>>]>,
    thread_id: usize,
}

impl Worker {
    pub fn spawn_thread(
        core_id: CoreId,
        thread_id: usize,
        threads_buffer: Arc<[spin::Mutex<ThreadBuffer<ComputeTask>>]>,
    ) {
        thread::Builder::new()
            .stack_size(resolve_stack_size())
            .spawn(move || {
                set_for_current(core_id);
                let worker = Worker::new(threads_buffer, thread_id);
                worker.work()
            })
            .unwrap();
    }
    fn new(
        threads_buffer: Arc<[spin::Mutex<ThreadBuffer<ComputeTask>>]>,
        thread_id: usize,
    ) -> Self {
        Self {
            threads_buffer,
            thread_id,
        }
    }
    fn work(self) {
        loop {
            if let Some(compute_task) = self.threads_buffer[self.thread_id].lock().pop() {
                compute_task.compute();
            }

            std::hint::spin_loop();
        }
    }
}
