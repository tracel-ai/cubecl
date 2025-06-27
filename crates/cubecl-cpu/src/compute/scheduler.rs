use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use cubecl_core::{CubeCount, ExecutionMode, compute::CubeTask, server::Bindings};

use crate::CpuCompiler;

use super::worker::Worker;

#[derive(Debug)]
pub struct Scheduler {
    threads: Vec<Worker>,
    stop: Arc<AtomicBool>,
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Scheduler {
    pub fn new() -> Scheduler {
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get())
            .into_iter()
            .map(|i| Worker::new(i, stop.clone()))
            .collect();

        Scheduler { threads, stop }
    }

    fn dispatch_execute(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
    ) {
    }
}
