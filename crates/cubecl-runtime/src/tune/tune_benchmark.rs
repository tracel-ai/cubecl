use super::{AutotuneError, TuneFn};
use crate::{client::ComputeClient, runtime::Runtime};
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;

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
        let client = self.client.clone();
        let name = self.operation.name().to_string();

        client
            .exclusive(move || self.profile_exclusive())
            .map_err(|err| AutotuneError::Unknown {
                name,
                err: err.to_string(),
            })?
    }

    fn profile_exclusive(self) -> Result<Vec<ProfileDuration>, AutotuneError> {
        self.warmup()?;

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
                        log::trace!("Error while autotuning {err:?}");
                        None
                    }
                },
                Err(err) => {
                    log::trace!("Error while autotuning {err:?}");
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

    fn warmup(&self) -> Result<(), AutotuneError> {
        let num_warmup = 3;

        let mut errors = Vec::with_capacity(num_warmup);
        // We make sure the server is in a correct state.
        let _errs = self.client.flush();

        for _ in 0..num_warmup {
            let op = self.operation.clone();
            let inputs = self.inputs.clone();
            let profiled = self
                .client
                .profile(move || op.execute(inputs), self.operation.name());

            match profiled {
                Ok(_) => {}
                Err(err) => errors.push(err),
            }
        }

        if errors.len() < num_warmup {
            Ok(())
        } else {
            let msg = alloc::format!("{:?}", errors.remove(num_warmup - 1));
            Err(AutotuneError::Unknown {
                name: self.operation.name().to_string(),
                err: msg,
            })
        }
    }
}
