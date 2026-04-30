use super::{AutotuneError, TuneFn, TuneInputs};
use crate::{client::ComputeClient, runtime::Runtime};
use alloc::string::ToString;
use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;

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

/// Benchmark how long this operation takes for a number of samples.
///
/// Returns at least one duration, otherwise an error is returned.
pub fn tune_benchmark<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    // `scoped` holds exclusive device access for the whole benchmark loop and
    // accepts non-`'static` closures.
    client
        .clone()
        .exclusive(move || profile_exclusive(operation, inputs, client))
        .map_err(|err| AutotuneError::Unknown {
            name: operation.name.to_string(),
            err: err.to_string(),
        })?
}

fn profile_exclusive<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    warmup(operation, inputs.clone(), client.clone())?;

    let num_samples = 10;
    let mut durations = Vec::new();

    for _ in 0..num_samples {
        let result: Result<
            (Result<Out, AutotuneError>, ProfileDuration),
            crate::server::ProfileError,
        > = {
            let inputs = inputs.clone();

            client.profile(
                move || {
                    // It is important to return the output since otherwise deadcode elimination
                    // might optimize away code that needs to be profiled.
                    operation.execute(inputs)
                },
                &operation.name,
            )
        };

        let result = match result {
            Ok((out, duration)) => match out {
                Ok(_) => Some(duration),
                Err(err) => {
                    log::trace!("Error while autotuning {err}");
                    None
                }
            },
            Err(err) => {
                log::trace!("Error while autotuning {err}");
                None
            }
        };

        if let Some(item) = result {
            durations.push(item);
        }
    }

    if durations.is_empty() {
        Err(AutotuneError::InvalidSamples {
            name: operation.name.to_string(),
        })
    } else {
        Ok(durations)
    }
}

fn warmup<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
) -> Result<(), AutotuneError> {
    let num_warmup = 3;

    let mut errors = Vec::with_capacity(num_warmup);
    // We make sure the server is in a correct state.
    let _errs = client.flush();

    for _ in 0..num_warmup {
        let inputs = inputs.clone();
        let profiled = client.profile(move || operation.execute(inputs), &operation.name);

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
            name: operation.name.to_string(),
            err: msg,
        })
    }
}
