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

impl<Out: Send + 'static> Clone for Box<dyn AutotuneOperation<Out>> {
    fn clone(&self) -> Self {
        self.as_ref().clone()
    }
}

impl<S: ComputeServer + 'static, C: ComputeChannel<S> + 'static, Out: Send + 'static>
    TuneBenchmark<S, C, Out>
{
    /// Benchmark how long this operation takes for a number of samples.
    pub async fn sample_durations(self) -> BenchmarkDurations {
        let operation = self.operation.clone();

        // If the inner operation need autotuning as well, we need to call it before.
        let _ = self.client.sync().await;
        operation.clone().execute();
        let _ = self.client.sync().await;

        let client = self.client.clone();

        let durations = self
            .client
            .profile(|| async move {
                let num_samples = 10;
                let mut durations = Vec::with_capacity(num_samples);

                for _ in 0..num_samples {
                    operation.clone().execute();
                    // For benchmarks - we need to wait for all tasks to complete before returning.
                    let duration = match client.sync_elapsed().await {
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
