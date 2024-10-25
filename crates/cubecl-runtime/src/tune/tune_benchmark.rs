use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use cubecl_common::benchmark::{BenchmarkDurations, TimingMethod};

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
        // If the inner operation need autotuning as well, we need to call it before.
        let _ = self.client.sync().await;
        AutotuneOperation::execute(self.operation.clone());
        let _ = self.client.sync().await;

        let durations = self
            .client
            .profile(|| async {
                let num_samples = 10;
                let mut durations = Vec::with_capacity(num_samples);

                for _ in 0..num_samples {
                    AutotuneOperation::execute(self.operation.clone());
                    // For benchmarks - we need to wait for all tasks to complete before returning.
                    let duration = match self.client.sync_elapsed().await {
                        Ok(val) => val,
                        Err(err) => {
                            #[cfg(not(target_family = "wasm"))]
                            panic!("Error while autotuning an operation: {:?}", err);

                            #[cfg(target_family = "wasm")]
                            {
                                // We can't panic inside a future on wasm.
                                log::warn!("Error while autotuning an operation: {:?}", err);
                                continue;
                            }
                        }
                    };
                    durations.push(duration);
                }
                durations
            })
            .await;

        BenchmarkDurations {
            timing_method: TimingMethod::DeviceOnly,
            durations,
        }
    }
}
