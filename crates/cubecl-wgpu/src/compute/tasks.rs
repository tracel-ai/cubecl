use std::{cell::RefCell, rc::Rc};

use crate::{WgpuResource, stream::WgpuStream};
use alloc::sync::Arc;
use cubecl_core::{CubeCount, ExecutionMode};
use cubecl_runtime::stream::scheduler::{SchedulerStreamBackend, SchedulerTask};

#[derive(Debug)]
pub(crate) enum LazyTask {
    Write {
        data: Vec<u8>,
        buffer: WgpuResource,
    },
    Execute {
        pipeline: Arc<wgpu::ComputePipeline>,
        count: CubeCount,
        resources: Vec<WgpuResource>,
    },
}

impl SchedulerTask for LazyTask {}

#[derive(new, Debug)]
pub(crate) struct ScheduledWgpuBackend {
    stream: Rc<RefCell<WgpuStream>>,
}

impl SchedulerStreamBackend for ScheduledWgpuBackend {
    type Task = LazyTask;

    fn execute(&mut self, tasks: impl Iterator<Item = Self::Task>) {
        let mut stream = self.stream.borrow_mut();

        for task in tasks {
            stream.execute_task(task);
        }
    }
}
