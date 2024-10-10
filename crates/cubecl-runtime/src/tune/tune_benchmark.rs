use cubecl_common::benchmark::Benchmark;
use futures_lite::future;

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;

use super::AutotuneOperation;
use alloc::boxed::Box;
use alloc::string::{String, ToString};

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<S: ComputeServer, C, Out = ()> {
    operation: Box<dyn AutotuneOperation<Out>>,
    client: ComputeClient<S, C>,
}

impl<Out> Clone for Box<dyn AutotuneOperation<Out>> {
    fn clone(&self) -> Self {
        self.as_ref().clone()
    }
}

impl<S: ComputeServer, C: ComputeChannel<S>, Out> Benchmark for TuneBenchmark<S, C, Out> {
    type Args = Box<dyn AutotuneOperation<Out>>;

    fn prepare(&self) -> Self::Args {
        self.operation.clone()
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, operation: Self::Args) {
        AutotuneOperation::execute(operation);
    }

    fn name(&self) -> String {
        "autotune".to_string()
    }

    fn sync(&self) {
        // For benchmarks - we need to wait for all tasks to complete before returning.
        future::block_on(self.client.sync());
    }
}
