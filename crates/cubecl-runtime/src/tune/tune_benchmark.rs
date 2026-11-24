use super::stage::{BoundStageEnterer, enter_stage};
use super::{AutotuneError, Evictor, TuneDeviceAccess, TuneFn, TuneInputs, TuneStage};
use crate::client::Client;
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

/// Benchmark how long this operation takes for a number of samples, inside a measuring stage
/// `stage` enters while the device is held.
///
/// Returns at least one duration, otherwise an error is returned.
pub(crate) fn tune_benchmark<'a, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: Client,
    evictor: Option<&mut Evictor<'_>>,
    stage: Option<&mut BoundStageEnterer<'_>>,
    access: TuneDeviceAccess,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    // Under exclusive access the device is held for the whole benchmark loop, which accepts
    // non-`'static` closures.
    let held = client.clone();
    access
        .hold(&held, move || {
            profile_exclusive(operation, inputs, client, evictor, stage, access)
        })
        .map_err(|err| AutotuneError::Unknown {
            name: operation.name.to_string(),
            err: err.to_string(),
        })?
}

fn profile_exclusive<'a, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: Client,
    mut evictor: Option<&mut Evictor<'_>>,
    stage: Option<&mut BoundStageEnterer<'_>>,
    access: TuneDeviceAccess,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    // These launches are the measurement, so their stream executes whatever
    // the process mode drops: it exists to skip the *workload*, not the tuning it is
    // there to provoke. The guard covers the warm-up too, since a candidate
    // measured without one is measured on its slowest run.
    //
    // Inside the `exclusive` call, so the stream switches to executing only
    // while this measurement holds the device. `exclusive` runs the body
    // under the issuing thread's stream, so the client resolves the same
    // stream here as there.
    let measuring =
        crate::execution::StreamModeOverride::new(crate::execution::StreamMode::Execute, &client);
    let measuring_stage = enter_stage(stage, TuneStage::Measuring);

    warmup(operation, inputs.clone(), client.clone(), access)?;

    // The same budget the adaptive scheduler reads. This pass takes the ceiling: with no
    // elimination, there is nothing for a smaller budget to buy, and a candidate that stops early
    // here would just be measured on less evidence than its rivals.
    let (_, num_samples) = crate::config::CubeClRuntimeConfig::get()
        .autotune
        .bench
        .sample_count_range();
    let mut durations = Vec::new();

    for _ in 0..num_samples {
        // A candidate that fails once is disqualified regardless of how the remaining samples
        // go, so the loop stops on the first error and hands it back untouched. Sampling on
        // would only pay more device round trips to reach the same verdict, with the reason
        // for the failure replaced by `InvalidSamples`.
        durations.push(operation.sample_once(
            inputs.clone(),
            &client,
            evictor.as_deref_mut(),
            access,
        )?);
    }
    core::mem::drop(measuring_stage);
    core::mem::drop(measuring);

    if durations.is_empty() {
        Err(AutotuneError::InvalidSamples {
            name: operation.name.to_string(),
        })
    } else {
        Ok(durations)
    }
}

fn warmup<'a, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: Client,
    access: TuneDeviceAccess,
) -> Result<(), AutotuneError> {
    let num_warmup = 3;

    let mut errors = Vec::with_capacity(num_warmup);
    // We make sure the server is in a correct state.
    let _errs = client.flush();

    for _ in 0..num_warmup {
        let inputs = inputs.clone();
        let profiled = access.profile(
            &client,
            || operation.race_contained(inputs, &client, access),
            &operation.name,
        );

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
        Err(operation.unprofiled(errors.remove(num_warmup - 1)))
    }
}
