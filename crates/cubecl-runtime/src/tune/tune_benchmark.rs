use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use cubecl_common::benchmark::BenchmarkDurations;

use super::AutotuneOperation;
use alloc::boxed::Box;

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

impl<S: ComputeServer, C: ComputeChannel<S>, Out> TuneBenchmark<S, C, Out> {
    /// Benchmark how long this operation takes for a number of samples.
    pub async fn sample_durations(&self) -> BenchmarkDurations {
        let num_samples = 10;

        let mut durations = vec![];
        for _ in 0..num_samples {
            self.client.sync().await;
            AutotuneOperation::execute(self.operation.clone());
            // For benchmarks - we need to wait for all tasks to complete before returning.
            let duration = self.client.sync().await;
            durations.push(duration);
        }
        BenchmarkDurations { durations }
    }
}
