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

/// Run the operation once without measuring it, to trigger compilation.
///
/// Expects to already hold exclusive device access; the adaptive driver takes it once for the
/// whole round robin rather than once per candidate.
pub(crate) fn warmup_once<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: &ComputeClient<R>,
) -> Result<(), AutotuneError> {
    let _errs = client.flush();

    // The inner result carries a failure to compile or launch, which is exactly what this call
    // exists to surface, so it is unwrapped rather than dropped along with the timing.
    let profiled = client.profile(move || operation.execute(inputs), &operation.name);

    match profiled {
        Ok((Ok(_), _)) => Ok(()),
        Ok((Err(err), _)) => Err(err),
        Err(err) => Err(AutotuneError::Unknown {
            name: operation.name.to_string(),
            err: err.to_string(),
        }),
    }
}

/// Queue a single measured execution. See [`warmup_once`] for the locking expectation.
pub(crate) fn sample_once<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: &ComputeClient<R>,
) -> Result<ProfileDuration, AutotuneError> {
    // The output is returned so dead code elimination can't drop the work being profiled.
    let profiled = client.profile(move || operation.execute(inputs), &operation.name);

    match profiled {
        Ok((Ok(_), duration)) => Ok(duration),
        Ok((Err(err), _)) => Err(err),
        Err(err) => Err(AutotuneError::Unknown {
            name: operation.name.to_string(),
            err: err.to_string(),
        }),
    }
}

fn profile_exclusive<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    // These launches are the measurement, so they run even inside a dry run:
    // that mode exists to skip the *workload*, not the tuning it is there to
    // provoke. The guard covers the warm-up too, since a candidate measured
    // without one is measured on its slowest run.
    //
    // It has to live here rather than around the `exclusive` call in
    // `tune_benchmark`: the guard is thread-local, and `exclusive` runs this
    // body on the device thread, which is where the launches below are issued
    // from.
    let _real_run = crate::dry_run::RealRun::new();

    warmup(operation, inputs.clone(), client.clone())?;

    let num_samples = 10;
    let mut durations = Vec::new();

    for _ in 0..num_samples {
        // A candidate that fails once is disqualified regardless of how the remaining samples
        // go, so the loop stops on the first error and hands it back untouched. Sampling on
        // would only pay more device round trips to reach the same verdict, with the reason
        // for the failure replaced by `InvalidSamples`.
        durations.push(sample_once(operation, inputs.clone(), &client)?);
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
            // The tunable rejected its own configuration, which it will do identically on
            // every call, so the remaining warmups and the whole sampling loop are skipped.
            // The error is propagated as-is to keep the reason it was rejected.
            Ok((Err(err), _)) => return Err(err),
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
