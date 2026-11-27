use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;
use cubecl_environment::config::RuntimeConfig;

use crate::tune::drive::candidate_sampler::{CandidateSampler, hold_failed};
use crate::tune::hook::binding::{BoundEvictor, BoundStageEnterer, enter_stage};
use crate::tune::set::tune_fn::TuneFn;
use crate::tune::{AutotuneCheckable, AutotuneError, TuneInputs, TuneStage};

/// The warmups the fixed-count pass runs before it samples a candidate.
const WARMUP_ROUND_COUNT: usize = 3;

/// Benchmark how long `operation` takes for the configured sample count, inside a measuring stage
/// `stage` enters while the device is held, its samples queued and not resolved.
///
/// # Guarantees
///
/// - Answers at least one sample: the configured count is at least one.
pub fn tune_benchmark<'a, F: TuneInputs, Out: AutotuneCheckable>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    sampler: CandidateSampler<'_>,
    evictor: Option<&mut BoundEvictor<'_>>,
    stage: Option<&mut BoundStageEnterer<'_>>,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    // Under exclusive access the device is held for the whole benchmark loop, which accepts
    // non-`'static` closures.
    sampler
        .hold(move || profile_held(operation, inputs, sampler, evictor, stage))
        .map_err(|err| hold_failed(operation.name(), &err))?
}

fn profile_held<'a, F: TuneInputs, Out: AutotuneCheckable>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    sampler: CandidateSampler<'_>,
    mut evictor: Option<&mut BoundEvictor<'_>>,
    stage: Option<&mut BoundStageEnterer<'_>>,
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
    let measuring = crate::execution::StreamModeOverride::new(
        crate::execution::StreamMode::Execute,
        sampler.client(),
    );
    let measuring_stage = enter_stage(stage, TuneStage::Measuring);

    sampler.warmup_rounds(operation, inputs.clone(), WARMUP_ROUND_COUNT)?;

    // The same budget the adaptive scheduler reads, at least one sample. This pass takes the
    // ceiling: with no elimination, there is nothing for a smaller budget to buy, and a candidate
    // that stops early here would just be measured on less evidence than its rivals.
    let (_, sample_count) = crate::config::CubeClRuntimeConfig::get()
        .autotune
        .bench
        .sample_count_range();

    // A candidate that fails once is disqualified regardless of how the remaining samples go, so
    // the loop stops on the first error and hands it back untouched: sampling on would only pay
    // more device round trips to reach the same verdict.
    let samples = (0..sample_count)
        .map(|_| sampler.sample_once(operation, inputs.clone(), evictor.as_deref_mut()))
        .collect();
    core::mem::drop(measuring_stage);
    core::mem::drop(measuring);
    samples
}
