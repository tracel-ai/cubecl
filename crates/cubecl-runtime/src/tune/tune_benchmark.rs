use super::{AutotuneError, TuneFn, TuneInputs};
use crate::{client::ComputeClient, runtime::Runtime};
use alloc::string::ToString;
use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;
use cubecl_environment::config::RuntimeConfig;

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

/// Calls `client.profile` for one candidate with a panic contained to that candidate.
///
/// The adaptive scheduler drives a whole round robin of candidates from a single `exclusive`
/// call, so a panic left to unwind — a compiler `unwrap` blowing up on a kernel the device
/// cannot compile, in practice — would fail every rival in the batch, not just the candidate
/// at fault. `client.profile` deliberately re-raises panics at its caller; tuning is the one
/// caller that must flatten them into a per-candidate error instead.
fn profile_candidate<O: Send + 'static, R: Runtime>(
    client: &ComputeClient<R>,
    name: &str,
    func: impl FnOnce() -> O + Send,
) -> Result<(O, ProfileDuration), AutotuneError> {
    let run = || {
        client
            .profile(func, name)
            .map_err(|err| AutotuneError::Unknown {
                name: name.to_string(),
                err: err.to_string(),
            })
    };

    #[cfg(feature = "std")]
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(payload) => {
            let err = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<alloc::string::String>().cloned())
                .unwrap_or_else(|| "candidate panicked with a non-string payload".to_string());
            Err(AutotuneError::Unknown {
                name: name.to_string(),
                err,
            })
        }
    }

    // Without std there is no unwinding to catch; a panic aborts either way.
    #[cfg(not(feature = "std"))]
    run()
}

impl<F: TuneInputs, Out: AutotuneOutput> TuneFn<F, Out> {
    /// Run the operation once without measuring it, to trigger compilation.
    ///
    /// Expects to already hold exclusive device access; the adaptive driver takes it once for
    /// the whole round robin rather than once per candidate.
    pub(crate) fn warmup_once<'a, R: Runtime>(
        &self,
        inputs: <F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
    ) -> Result<(), AutotuneError> {
        // We make sure the server is in a correct state.
        let _errs = client.flush();

        // The profile is dropped without being resolved: a warmup only exists to surface a
        // failure to compile or launch, which is what the error carries.
        self.sample_once(inputs, client).map(|_| ())
    }

    /// Queue a single measured execution. See [`Self::warmup_once`] for the locking expectation.
    pub(crate) fn sample_once<'a, R: Runtime>(
        &self,
        inputs: <F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
    ) -> Result<ProfileDuration, AutotuneError> {
        // The output is returned so dead code elimination can't drop the work being profiled.
        let profiled = profile_candidate(client, &self.name, move || self.execute(inputs));

        match profiled {
            Ok((Ok(_), duration)) => Ok(duration),
            Ok((Err(err), _)) => Err(err),
            Err(err) => Err(err),
        }
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

    // The same budget the adaptive scheduler reads. This pass takes the ceiling: with no
    // elimination, there is nothing for a smaller budget to buy, and a candidate that stops early
    // here would just be measured on less evidence than its rivals.
    let (_, num_samples) = crate::config::CubeClRuntimeConfig::get()
        .autotune
        .bench
        .samples();
    let mut durations = Vec::new();

    for _ in 0..num_samples {
        // A candidate that fails once is disqualified regardless of how the remaining samples
        // go, so the loop stops on the first error and hands it back untouched. Sampling on
        // would only pay more device round trips to reach the same verdict, with the reason
        // for the failure replaced by `InvalidSamples`.
        durations.push(operation.sample_once(inputs.clone(), &client)?);
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
        let profiled =
            profile_candidate(&client, &operation.name, move || operation.execute(inputs));

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
