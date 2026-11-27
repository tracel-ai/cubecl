use alloc::string::ToString;
use cubecl_common::profile::ProfileDuration;
#[cfg(feature = "std")]
use cubecl_environment::backtrace::BackTrace;

use crate::client::{Client, ProfileWindow};
use crate::server::{ProfileError, ServerError};
use crate::tune::hook::binding::BoundEvictor;
use crate::tune::set::tune_fn::TuneFn;
use crate::tune::{AutotuneError, TuneDeviceAccess, TuneInputs};

/// Runs a tune's candidates on one device, the way its [`TuneDeviceAccess`] holds the device:
/// warms them up, takes their samples, and contains their panics.
///
/// # Assumptions
///
/// - Every function but [`hold`](Self::hold) and [`client`](Self::client) runs inside a
///   [`hold`](Self::hold), so that only a panic stops a race: the adaptive driver takes it once
///   for the whole round robin rather than once per candidate.
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
    pub fn client(&self) -> &'c Client {
        self.client
    }

    /// What `task` returns, run with the device held where the access holds it.
    ///
    /// # Errors
    ///
    /// Where the device could not be taken exclusively.
    pub fn hold<'t, R: Send + 'static>(
        &self,
        task: impl FnOnce() -> R + Send + 't,
    ) -> Result<R, ServerError>
    where
        'c: 't,
    {
        match self.access {
            TuneDeviceAccess::Exclusive => self.client.exclusive(task),
            TuneDeviceAccess::Shared => Ok(task()),
        }
    }

    /// Race `operation` on `inputs` ([`TuneFn::race`]), its panic contained so it costs no other
    /// candidate its measurement: in a device call of its own where the device is held, on the
    /// calling thread where it never is.
    pub fn contain_race<'a, I: TuneInputs, Out: Send + 'static>(
        &self,
        operation: &TuneFn<I, Out>,
        inputs: <I as TuneInputs>::At<'a>,
    ) -> Result<Out, AutotuneError> {
        self.contained_race(operation.name(), || operation.race(inputs))
    }

    #[cfg(not(target_family = "wasm"))]
    /// Run `operation` once without measuring it, to trigger compilation.
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
            let inputs = inputs.clone();
            match self.profile(
                move || self.contain_race(operation, inputs),
                operation.name(),
            ) {
                // The tunable rejected its own configuration, which it will do identically on
                // every call, so the remaining warmups and the whole sampling loop are skipped.
                // The error is propagated as-is to keep the reason it was rejected.
                Ok((Err(err), _)) => return Err(err),
                Ok(_) => profiled = true,
                Err(err) => last_failure = Some(err),
            }
        }

        match last_failure.filter(|_| !profiled) {
            Some(err) => Err(logged(AutotuneError::unprofiled(operation.name(), err))),
            None => Ok(()),
        }
    }

    /// Queue a single measured execution.
    ///
    /// `evictor` runs first, outside the profiled region, so the sample reads what a real call
    /// reads rather than what the previous launch left in cache ([`Evictor`](crate::tune::Evictor)). An
    /// eviction that fails is logged and the sample taken warm: the measurement is still worth
    /// more than none, and the failure is the eviction's, not the candidate's.
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
        let profiled = self.profile(
            move || self.contain_race(operation, inputs),
            operation.name(),
        );
        sampled(operation.name(), profiled)
    }
}

impl CandidateSampler<'_> {
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

    /// What `func` returns and how long its launches took on the calling stream.
    fn profile<O: Send + 'static>(
        &self,
        func: impl FnOnce() -> O + Send,
        name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        match self.access {
            TuneDeviceAccess::Exclusive => self.client.profile(func, name),
            TuneDeviceAccess::Shared => profile_on_stream(self.client, func),
        }
    }
}

/// What a profiled sample of the candidate `name` answers: its duration, its own error, or what a
/// failed profile says of it.
fn sampled<Out>(
    name: &str,
    profiled: Result<(Result<Out, AutotuneError>, ProfileDuration), ProfileError>,
) -> Result<ProfileDuration, AutotuneError> {
    match profiled {
        Ok((Ok(_), duration)) => Ok(duration),
        Ok((Err(err), _)) => Err(err),
        Err(err) => Err(logged(AutotuneError::unprofiled(name, err))),
    }
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
            sampled::<()>("candidate", Ok((Err(rejection), unresolved()))),
            Err(AutotuneError::Rejected { .. })
        ));
    }

    /// A profile that resolves to nothing, for samples that are never resolved.
    fn unresolved() -> ProfileDuration {
        ProfileDuration::new_system_time(
            cubecl_common::profile::Instant::now(),
            cubecl_common::profile::Instant::now(),
        )
    }
}
