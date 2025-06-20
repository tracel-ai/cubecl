use core::time::Duration;

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use cubecl_common::benchmark::ProfileDuration;

use super::{AutotuneError, Tunable};

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<S: ComputeServer, C, In: Clone + Send + 'static, Out: Send + 'static> {
    operation: Arc<dyn Tunable<Inputs = In, Output = Out>>,
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
    #[cfg(feature = "autotune-checks")]
    pub(crate) fn output_for_checks(&self) -> Result<Out, AutotuneError> {
        self.operation.clone().execute(self.inputs.clone())
    }

    /// Benchmark how long this operation takes for a number of samples.
    pub fn profile(self) -> Result<Vec<ProfileDuration>, AutotuneError> {
        let operation = self.operation;
        // If the inner operation need autotuning as well, we need to call it before. This will
        // recurse and keep calling operations until a leaf operation tunes, and so on. This effectively
        // does a depth-first traversal of the operation tree. Without this, client.profile() would have to
        // support profiling recursively.
        let mut error = None;

        // For now we wrap the warmup operation inside a profiling task, since for now we have
        // basic error handling for such task that may help catch ressources errors.
        let result = self
            .client
            .profile(|| match operation.execute(self.inputs.clone()) {
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                }
            });

        if let Err(err) = result {
            return Err(AutotuneError::Unknown(format!("{err:?}")));
        };

        if let Some(err) = error {
            return Err(err);
        };

        let num_samples = 10;
        let durations = (0..num_samples)
            .map(|_| {
                let result: Result<ProfileDuration, crate::server::ProfileError> =
                    self.client.profile(|| {
                        // It is important to return the output since otherwise deadcode elimination
                        // might optimize away code that needs to be profiled.
                        operation
                            .execute(self.inputs.clone())
                            .expect("Should not fail when previously tried during the warmup.")
                    });
                match result {
                    Ok(val) => val,
                    Err(err) => {
                        log::warn!("Error while autotuning {err:?}");
                        ProfileDuration::from_duration(Duration::from_millis(u64::MAX))
                    }
                }
            })
            .collect();

        Ok(durations)
    }
}
