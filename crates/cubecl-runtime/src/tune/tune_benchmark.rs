use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::time::Duration;
use cubecl_common::profile::{ProfileDuration, TimingMethod};

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;

use super::{AutotuneError, TuneFn};

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<S: ComputeServer, C, In: Clone + Send + 'static, Out: Send + 'static> {
    operation: Arc<dyn TuneFn<Inputs = In, Output = Out>>,
    inputs: In,
    client: ComputeClient<S, C>,
}

/// The trait to be implemented by an autotune output.
pub trait AutotuneOutput: Send + 'static {
    #[cfg(feature = "autotune-checks")]
    /// Checks if the output of an autotune operation is the same as another one on the same
    /// problem.
    fn check_equivalence(&self, other: Self);
}

impl AutotuneOutput for () {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, _other: Self) {
        //
    }
}

impl<
    S: ComputeServer + 'static,
    C: ComputeChannel<S> + 'static,
    In: Clone + Send + 'static,
    Out: AutotuneOutput,
> TuneBenchmark<S, C, In, Out>
{
    /// Benchmark how long this operation takes for a number of samples.
    ///
    /// Returns at least one duration, otherwise an error is returned.
    pub async fn profile(self) -> Result<(Vec<Duration>, TimingMethod), AutotuneError> {
        let operation = self.operation;
        let num_samples = 10;
        let mut durations: Vec<_> = Vec::with_capacity(num_samples);
        let mut timing = TimingMethod::System;

        for i in 0..num_samples + 1 {
            let mut error = None;
            let result: Result<ProfileDuration, crate::server::ProfileError> = self.client.profile(
                || {
                    // It is important to return the output since otherwise deadcode elimination
                    // might optimize away code that needs to be profiled.
                    match operation.execute(self.inputs.clone()) {
                        Ok(val) => Ok(val),
                        Err(err) => {
                            error = Some(err);
                            Err(())
                        }
                    }
                },
                operation.name(),
            );

            match result {
                Ok(val) => {
                    timing = val.timing_method();
                    let duration = val.resolve().await.duration();

                    // We need to await the duration before.
                    if let Some(err) = error {
                        return Err(err);
                    }

                    // We skip the first result, since it acts as a warmup step.
                    if i > 0 {
                        durations.push(duration);
                    }
                }
                Err(err) => {
                    log::warn!("Error while autotuning {err:?}");
                    return Err(AutotuneError::Unknown(format!("{err:?}")));
                }
            }
        }

        if durations.is_empty() {
            Err(AutotuneError::InvalidSamples)
        } else {
            Ok((durations, timing))
        }
    }
}
