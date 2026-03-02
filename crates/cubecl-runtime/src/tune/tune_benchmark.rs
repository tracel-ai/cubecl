use super::{AutotuneError, TuneFn};
use crate::{client::ComputeClient, runtime::Runtime};
use alloc::format;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::profile::{ProfileDuration, TimingMethod};

/// A benchmark that runs on server handles
#[derive(new)]
pub struct TuneBenchmark<R: Runtime, In: Clone + Send + 'static, Out: Send + 'static> {
    operation: Arc<dyn TuneFn<Inputs = In, Output = Out>>,
    inputs: In,
    client: ComputeClient<R>,
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

impl<R: Runtime, In: Clone + Send + 'static, Out: AutotuneOutput> TuneBenchmark<R, In, Out> {
    /// Benchmark how long this operation takes for a number of samples.
    ///
    /// Returns at least one duration, otherwise an error is returned.
    pub fn profile(self) -> Result<Vec<ProfileDuration>, AutotuneError> {
        // If the inner operation need autotuning as well, we need to call it before. This will
        // recurse and keep calling operations until a leaf operation tunes, and so on. This effectively
        // does a depth-first traversal of the operation tree.

        // For now we wrap the warmup operation inside a profiling task, since we have basic error
        // handling for system timing methods.
        match self.client.properties().timing_method {
            TimingMethod::System => self.warmup_full_error_handling(),
            TimingMethod::Device => self.warmup_minimal_error_handling(),
        }?;

        let operation = self.operation.clone();
        let name = operation.name().to_string();
        let num_samples = 10;
        let mut durations = Vec::new();
        for _ in 0..num_samples {
            let result: Result<
                (Result<Out, AutotuneError>, ProfileDuration),
                crate::server::ProfileError,
            > = {
                let inputs = self.inputs.clone();
                let operation = operation.clone();

                self.client.profile(
                    move || {
                        // It is important to return the output since otherwise deadcode elimination
                        // might optimize away code that needs to be profiled.
                        operation.execute(inputs)
                    },
                    &name,
                )
            };

            let result = match result {
                Ok((out, duration)) => match out {
                    Ok(_) => Some(duration),
                    Err(err) => {
                        log::warn!("Error while autotuning {err:?}");
                        None
                    }
                },
                Err(err) => {
                    log::warn!("Error while autotuning {err:?}");
                    None
                }
            };

            if let Some(item) = result {
                durations.push(item);
            }
        }

        if durations.is_empty() {
            Err(AutotuneError::InvalidSamples { name })
        } else {
            Ok(durations)
        }
    }

    fn warmup_full_error_handling(&self) -> Result<(), AutotuneError> {
        let error = Arc::new(spin::Mutex::new(None));

        let operation = self.operation.clone();
        let inputs = self.inputs.clone();
        let error_cloned = error.clone();
        let result = self.client.profile(
            move || {
                if let Err(err) = operation.execute(inputs) {
                    let mut error = error_cloned.lock();
                    *error = Some(err);
                }
            },
            self.operation.name(),
        );

        if let Err(err) = result {
            return Err(AutotuneError::Unknown {
                name: self.operation.name().to_string(),
                err: format!("{err:?}"),
            });
        };

        if let Some(err) = error.lock().as_ref() {
            return Err(err.clone());
        };

        Ok(())
    }
    fn warmup_minimal_error_handling(&self) -> Result<(), AutotuneError> {
        self.operation.execute(self.inputs.clone())?;
        Ok(())
    }
}
