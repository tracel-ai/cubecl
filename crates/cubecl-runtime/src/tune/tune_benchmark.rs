use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use cubecl_common::benchmark::{BenchmarkDurations, TimingMethod};

use super::{AutotuneError, Tunable};

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<S: ComputeServer, C, In: Clone + Send + 'static, Out: Send + 'static> {
    operation: Arc<dyn Tunable<Inputs = In, Output = Out>>,
    inputs: In,
    client: ComputeClient<S, C>,
}

impl<
    S: ComputeServer + 'static,
    C: ComputeChannel<S> + 'static,
    In: Clone + Send + 'static,
    Out: Send + 'static,
> TuneBenchmark<S, C, In, Out>
{
    /// Benchmark how long this operation takes for a number of samples.
    pub async fn sample_durations(self) -> Result<BenchmarkDurations, AutotuneError> {
        let operation = self.operation;

        // If the inner operation need autotuning as well, we need to call it before.
        let _ = self.client.sync().await;
        operation.clone().execute(self.inputs.clone())?;

        let _ = self.client.sync().await;

        let client = self.client.clone();

        let durations = self
            .client
            .profile(|| async move {
                let num_samples = 10;
                let mut durations = Vec::with_capacity(num_samples);

                for _ in 0..num_samples {
                    operation
                        .clone()
                        .execute(self.inputs.clone())
                        .expect("Should not fail when previously tried during the warmup.");
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

        Ok(BenchmarkDurations {
            timing_method: TimingMethod::DeviceOnly,
            durations,
        })
    }
}
