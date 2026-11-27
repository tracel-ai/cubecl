use alloc::string::ToString;
use cubecl_common::profile::ProfileDuration;
#[cfg(feature = "std")]
use cubecl_environment::backtrace::BackTrace;

use cubecl_environment::sync::{AtomicBool, Ordering};

use crate::client::{Client, ProfileWindow};
use crate::execution::{StreamMode, StreamModeOverride};
use crate::server::{ProfileError, ServerError};
use crate::tune::hook::hook_binder::{BoundEvictor, BoundStageEnterer, enter_stage};
use crate::tune::measure::tune_fn::TuneFn;
use crate::tune::{AutotuneError, TuneDeviceAccess, TuneInputs, TuneStage};
use alloc::boxed::Box;
use core::any::Any;

/// A measuring stage open on a sampler's stream until it drops: the stream executes its launches,
/// whatever the process mode discards, and the set's enterer entered the measuring stage.
pub struct MeasuringStage {
    // Dropped first: the stage ends before its launches stop executing.
    _stage: Option<Box<dyn Any>>,
    _executing: StreamModeOverride,
}

/// Runs a tune's candidates on one device, the way its [`TuneDeviceAccess`] holds the device:
/// warms them up, takes their samples, and contains their panics.
#[derive(Clone, Copy)]
pub struct CandidateSampler<'c> {
    client: &'c Client,
    access: TuneDeviceAccess,
}

impl<'c> CandidateSampler<'c> {
    pub fn new(client: &'c Client, access: TuneDeviceAccess) -> Self {
        Self { client, access }
    }

    /// The client of the device the candidates run on.
    #[cfg(any(not(target_family = "wasm"), feature = "autotune-checks"))]
    pub fn client(&self) -> &'c Client {
        self.client
    }

    /// What `task` returns, run with the device held where the access holds it.
    ///
    /// # Errors
    ///
    /// Where the device could not be taken exclusively. Nothing ran.
    ///
    /// # Panics
    ///
    /// Where `task` panics outside a [contained race](Self::contain_race): only a candidate's own
    /// panic is contained.
    pub fn hold<'t, R: Send + 'static>(
        &self,
        task: impl FnOnce() -> R + Send + 't,
    ) -> Result<R, ServerError>
    where
        'c: 't,
    {
        match self.access {
            TuneDeviceAccess::Exclusive => self.hold_exclusively(task),
            TuneDeviceAccess::Shared => Ok(task()),
        }
    }

    /// Open the measuring stage `enterer` enters, the launches of the measurement executing.
    ///
    /// # Assumptions
    ///
    /// - It runs inside a [`hold`](Self::hold): the stream switches to executing only while the
    ///   measurement holds the device.
    pub fn open_measuring_stage(
        &self,
        enterer: Option<&mut BoundStageEnterer<'_>>,
    ) -> MeasuringStage {
        let executing = StreamModeOverride::new(StreamMode::Execute, self.client);
        MeasuringStage {
            _stage: enter_stage(enterer, TuneStage::Measuring),
            _executing: executing,
        }
    }

    /// Race `operation` on `inputs` ([`TuneFn::race`]), its panic contained so it costs no other
    /// candidate its measurement: in a device call of its own where the device is held, on the
    /// calling thread where it never is.
    ///
    /// # Assumptions
    ///
    /// - It runs inside a [`hold`](Self::hold), so that only a panic stops the race.
    pub fn contain_race<'a, I: TuneInputs, Out: Send + 'static>(
        &self,
        operation: &TuneFn<I, Out>,
        inputs: <I as TuneInputs>::At<'a>,
    ) -> Result<Out, AutotuneError> {
        self.contained_race(operation.name(), || operation.race(inputs))
    }

    /// Run `operation` once without measuring it, to trigger compilation.
    ///
    /// # Assumptions
    ///
    /// - It runs inside a [`hold`](Self::hold): the adaptive driver takes one for the whole round
    ///   robin rather than one per candidate.
    #[cfg(not(target_family = "wasm"))]
    pub fn warmup_once<'a, I: TuneInputs, Out: Send + 'static>(
        &self,
        operation: &TuneFn<I, Out>,
        inputs: <I as TuneInputs>::At<'a>,
    ) -> Result<(), AutotuneError> {
        // We make sure the server is in a correct state.
        let _errs = self.client.flush();

        // The profile is dropped without being resolved: a warmup only exists to surface a
        // failure to compile or launch, which is what the error carries — so nothing is evicted
        // for it, either.
        self.sample_once(operation, inputs, None).map(|_| ())
    }

    /// Run `operation` `round_count` times, each profiled and none resolved, to trigger
    /// compilation and settle the clocks, as the fixed-count pass warms a candidate up.
    ///
    /// # Assumptions
    ///
    /// - It runs inside a [`hold`](Self::hold).
    ///
    /// # Errors
    ///
    /// The candidate's own error on the round it returned one, which it would return on every
    /// round, or what the last profile says of it where every round's profile failed.
    pub fn warmup_rounds<'a, I: TuneInputs, Out: Send + 'static>(
        &self,
        operation: &TuneFn<I, Out>,
        inputs: <I as TuneInputs>::At<'a>,
        round_count: usize,
    ) -> Result<(), AutotuneError> {
        // We make sure the server is in a correct state.
        let _errs = self.client.flush();
        let mut profiled = false;
        let mut last_failure = None;

        for _ in 0..round_count {
            match self.profiled_race(operation, inputs.clone()) {
                // The tunable rejected its own configuration, which it will do identically on
                // every call, so the remaining warmups and the whole sampling loop are skipped.
                // The error is propagated as-is to keep the reason it was rejected.
                Ok((Err(err), _)) => return Err(err),
                Ok(_) => profiled = true,
                Err(err) => last_failure = Some(err),
            }
        }

        last_failure.filter(|_| !profiled).map_or(Ok(()), Err)
    }

    /// Queue a single measured execution.
    ///
    /// `evictor` runs first, outside the profiled region, so the sample reads what a real call
    /// reads rather than what the previous launch left in cache ([`Evictor`](crate::tune::Evictor)). An
    /// eviction that fails is logged and the sample taken warm: the measurement is still worth
    /// more than none, and the failure is the eviction's, not the candidate's.
    ///
    /// # Assumptions
    ///
    /// - It runs inside a [`hold`](Self::hold).
    pub fn sample_once<'a, I: TuneInputs, Out: Send + 'static>(
        &self,
        operation: &TuneFn<I, Out>,
        inputs: <I as TuneInputs>::At<'a>,
        evictor: Option<&mut BoundEvictor<'_>>,
    ) -> Result<ProfileDuration, AutotuneError> {
        if let Some(evict) = evictor
            && let Err(err) = evict()
        {
            log::error!(
                "The eviction before a sample of `{}` failed, so the sample is measured on a warm cache.\n{err}",
                operation.name()
            );
        }
        // The output is returned so dead code elimination can't drop the work being profiled.
        sampled(self.profiled_race(operation, inputs))
    }
}

impl CandidateSampler<'_> {
    /// [`hold`](Self::hold) under exclusive access: a hold that fails once `task` started is
    /// `task`'s own panic, raised again.
    fn hold_exclusively<'t, R: Send + 'static>(
        &self,
        task: impl FnOnce() -> R + Send + 't,
    ) -> Result<R, ServerError> {
        let started = AtomicBool::new(false);
        let running = &started;
        let held = self.client.exclusive(move || {
            running.store(true, Ordering::Relaxed);
            task()
        });
        assert!(
            held.is_ok() || !started.load(Ordering::Relaxed),
            "a task held on the device should not panic outside a candidate's race: {:?}",
            held.as_ref().err()
        );
        held
    }

    /// The race of `operation` on `inputs`, contained, and how long its launches took.
    fn profiled_race<'a, I: TuneInputs, Out: Send + 'static>(
        &self,
        operation: &TuneFn<I, Out>,
        inputs: <I as TuneInputs>::At<'a>,
    ) -> Result<(Result<Out, AutotuneError>, ProfileDuration), AutotuneError> {
        self.profile(
            move || self.contain_race(operation, inputs),
            operation.name(),
        )
    }

    /// What the race of the candidate `name` answers, its panic contained.
    fn contained_race<Out: Send + 'static>(
        &self,
        name: &str,
        race: impl FnOnce() -> Result<Out, AutotuneError> + Send,
    ) -> Result<Out, AutotuneError> {
        self.contained(race).unwrap_or_else(|panic| {
            Err(logged(AutotuneError::Panicked {
                name: name.into(),
                message: panic.to_string(),
            }))
        })
    }

    /// What `task` returns, its panic contained as the access contains it.
    fn contained<R: Send + 'static>(
        &self,
        task: impl FnOnce() -> R + Send,
    ) -> Result<R, ServerError> {
        match self.access {
            TuneDeviceAccess::Exclusive => self.client.exclusive(task),
            TuneDeviceAccess::Shared => caught(task),
        }
    }

    /// What `func`, run for the candidate `name`, returns and how long its launches took on the
    /// calling stream, or what the failed profile says of the candidate.
    fn profile<O: Send + 'static>(
        &self,
        func: impl FnOnce() -> O + Send,
        name: &str,
    ) -> Result<(O, ProfileDuration), AutotuneError> {
        let started = AtomicBool::new(false);
        let running = &started;
        let func = move || {
            running.store(true, Ordering::Relaxed);
            func()
        };
        let profiled = match self.access {
            TuneDeviceAccess::Exclusive => self.client.profile(func, name),
            TuneDeviceAccess::Shared => profile_on_stream(self.client, func),
        };
        profiled.map_err(|error| {
            if started.load(Ordering::Relaxed) {
                logged(AutotuneError::unprofiled(name, error))
            } else {
                AutotuneError::unopened(name, &error)
            }
        })
    }
}

/// What a profiled sample answers: its duration, its own error, or what its failed profile says
/// of it.
fn sampled<Out>(
    profiled: Result<(Result<Out, AutotuneError>, ProfileDuration), AutotuneError>,
) -> Result<ProfileDuration, AutotuneError> {
    profiled.and_then(|(output, duration)| output.map(|_| duration))
}

/// `error`, logged first where it is a panic: a defect in the candidate.
fn logged(error: AutotuneError) -> AutotuneError {
    if let AutotuneError::Panicked { name, message } = &error {
        log::error!(
            "The autotune candidate `{name}` panicked and is dropped from its race: {message}"
        );
    }
    error
}

/// What `task` returns, or the panic it stopped on. Without `std` a panic is never caught.
fn caught<R>(task: impl FnOnce() -> R) -> Result<R, ServerError> {
    #[cfg(feature = "std")]
    {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)).map_err(|payload| {
            ServerError::Generic {
                reason: crate::panic_message::panic_message(&*payload),
                backtrace: BackTrace::capture(),
            }
        })
    }
    #[cfg(not(feature = "std"))]
    {
        Ok(task())
    }
}

/// `func` timed between a window opened and closed on `client`'s calling stream, the window
/// abandoned where `func` unwinds.
fn profile_on_stream<O>(
    client: &Client,
    func: impl FnOnce() -> O,
) -> Result<(O, ProfileDuration), ProfileError> {
    let mut open = OpenWindow {
        client,
        window: Some(client.profile_start()?),
    };
    let out = func();
    let window = open.take();
    client.profile_end(window).map(|duration| (out, duration))
}

/// A window opened on a client's stream, abandoned on drop unless it was taken to be closed.
struct OpenWindow<'a> {
    client: &'a Client,
    window: Option<ProfileWindow>,
}

impl OpenWindow<'_> {
    /// The window, to be closed by its taker.
    fn take(&mut self) -> ProfileWindow {
        self.window
            .take()
            .expect("an open window should be taken once")
    }
}

impl Drop for OpenWindow<'_> {
    fn drop(&mut self) {
        if let Some(window) = self.window.take() {
            self.client.profile_abandon(window);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn a_caught_panic_answers_with_its_message() {
        let caught = caught(|| panic!("the block divides the vector"));

        assert!(
            matches!(caught, Err(ServerError::Generic { reason, .. }) if reason == "the block divides the vector")
        );
    }

    #[test]
    fn a_sample_the_candidate_rejected_keeps_its_rejection() {
        let rejection = AutotuneError::Rejected {
            name: "candidate".into(),
            reason: "no line of this shape".into(),
        };
        assert!(matches!(
            sampled::<()>(Ok((Err(rejection), unresolved()))),
            Err(AutotuneError::Rejected { .. })
        ));
    }

    /// A profile of a zero duration, for samples that are never resolved.
    fn unresolved() -> ProfileDuration {
        ProfileDuration::new_system_time(
            cubecl_common::profile::Instant::now(),
            cubecl_common::profile::Instant::now(),
        )
    }
}
