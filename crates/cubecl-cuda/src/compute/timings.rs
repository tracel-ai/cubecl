//! Device profiling for CUDA, on the GPU's own clock.
//!
//! A profiling window is two [events](cudarc::driver::sys::CUevent) recorded
//! into the stream where the window opened and closed. Nothing on the host
//! waits for them: the device stamps each one as the queue reaches it, and the
//! span between the two is read back later, by whoever awaits the measurement.
//!
//! That deferral is what makes a measurement mean something when profiles
//! nest. The [system-time
//! profiler](cubecl_runtime::timestamp_profiler::TimestampProfiler) has to
//! drain the stream at both ends of every window to have anything to time, so
//! an inner window's two drains happen *inside* the outer one and are charged
//! to it: the outer measurement grows with the number of inner ones, and every
//! window reads back the launch latency the drain exposed rather than the work.
//! Events are recorded in-queue and cost the host nothing, so an inner window
//! measures its own kernels and the outer one measures the work rather than the
//! profiling.
//!
//! The HIP backend's `EventProfiler` is the same design over `hipEvent_t`; the
//! two are meant to stay in step.

use std::sync::{Arc, Mutex};

use cubecl_common::profile::{Duration, Instant, ProfileDuration, ProfileTicks};
use cubecl_core::server::{ProfileError, ProfilingToken, ServerError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cudarc::driver::DriverError;
use cudarc::driver::result::{event, stream};
use cudarc::driver::sys::{CUevent, CUevent_flags, CUstream};

/// How long an [anchor](Anchor) is trusted before the next window takes a fresh
/// one.
///
/// `cuEventElapsedTime` answers in `f32` milliseconds, so the further an event
/// sits from the anchor the coarser its placement on the host clock: a second
/// out, the representable step is about 60 ns; an hour out, a quarter of a
/// millisecond. A window's own duration is measured between its own two events
/// and is unaffected — this only bounds the error on *where* the window lands,
/// which is what a tracing profiler lines its spans up with.
const ANCHOR_MAX_AGE: Duration = Duration::from_secs(1);

/// What a window reports when its events could not be read back.
///
/// The read only fails when the device is already in trouble, and a resolved
/// profile is a duration rather than a result — there is no error to hand back
/// at that point. Reporting zero would give an autotune sweep a candidate that
/// won by failing, so an unreadable window reports a time no real one reaches
/// and says why in the log.
const UNREADABLE: Duration = Duration::from_secs(3600);

/// The device profiler: the windows currently open, and the anchor that places
/// them on the host clock.
#[derive(Debug, Default)]
pub struct EventProfiler {
    open: HashMap<ProfilingToken, Result<Open, ProfileError>>,
    counter: u64,
    pool: EventPool,
    /// Created with the first window rather than with the profiler: a process
    /// that never profiles should not own a stream and an event for it.
    anchoring: Option<Anchoring>,
}

/// A window that has opened and not yet closed.
#[derive(Debug)]
struct Open {
    start: Event,
    /// The anchor this window opened under, kept so that its start and its end
    /// are placed against the same one however many refreshes happen in
    /// between.
    anchor: Anchor,
}

impl EventProfiler {
    /// Open a window at the current position of `stream`.
    pub fn start(&mut self, stream: CUstream) -> Result<ProfilingToken, ServerError> {
        let anchor = self.anchor().map_err(server_error)?;
        let start = self.pool.acquire().map_err(server_error)?;
        start.record(stream).map_err(server_error)?;

        let token = ProfilingToken { id: self.counter };
        self.counter += 1;
        self.open.insert(token, Ok(Open { start, anchor }));

        Ok(token)
    }

    /// Close the window `token` opened at the current position of `stream`.
    ///
    /// Returns at enqueue time. The duration is read from the device by the
    /// returned future, which is where the wait for the window's work lives.
    pub fn stop(
        &mut self,
        stream: CUstream,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        let Open { start, anchor } = match self.open.remove(&token) {
            Some(state) => state?,
            None => {
                return Err(ProfileError::NotRegistered {
                    backtrace: BackTrace::capture(),
                });
            }
        };

        let end = self.pool.acquire().map_err(profile_error)?;
        end.record(stream).map_err(profile_error)?;

        let pool = self.pool.clone();

        Ok(ProfileDuration::new_device_time(async move {
            let ticks = read(&start, &end, &anchor).unwrap_or_else(|err| {
                log::error!(
                    "Could not read back a CUDA profiling window ({err}); reporting {UNREADABLE:?} \
                     so nothing mistakes it for a fast one"
                );
                let now = Instant::now();
                ProfileTicks::from_start_end(now, now + UNREADABLE)
            });

            pool.release(start);
            pool.release(end);

            ticks
        }))
    }

    /// Drop the window `token` opened without measuring it, for a caller that
    /// has no way to record its end.
    pub fn abandon(&mut self, token: ProfilingToken) {
        self.open.remove(&token);
    }

    /// Register an error against every open window.
    pub fn error(&mut self, error: ProfileError) {
        self.open
            .iter_mut()
            .for_each(|(_, state)| *state = Err(error.clone()));
    }

    /// Mark every open window invalid because device work failed.
    ///
    /// This is what keeps a tuning candidate that failed from benchmarking at
    /// close to zero and winning the tune. A no-op with no window open, so a
    /// failure path calls it unconditionally and pays nothing for the common
    /// case of no measurement in flight.
    pub fn failure(&mut self, error: &ServerError) {
        if self.open.is_empty() {
            return;
        }
        self.error(error.into());
    }

    /// The anchor a window opening now measures against, refreshed when the
    /// current one has aged past [`ANCHOR_MAX_AGE`].
    fn anchor(&mut self) -> Result<Anchor, DriverError> {
        if self.anchoring.is_none() {
            self.anchoring = Some(Anchoring::new(&self.pool)?);
        }
        let anchoring = self.anchoring.as_mut().expect("filled right above");

        if anchoring.current.instant.elapsed() > ANCHOR_MAX_AGE {
            anchoring.current = Anchor::take(anchoring.stream, &self.pool)?;
        }

        Ok(anchoring.current.clone())
    }
}

/// Read a closed window back from the device.
fn read(start: &Event, end: &Event, anchor: &Anchor) -> Result<ProfileTicks, DriverError> {
    // Both, rather than the end alone. The end implies the start only because
    // the two are recorded on the same stream, and waiting on an event the
    // device has already passed returns immediately — a free way to stop
    // depending on that.
    start.wait()?;
    end.wait()?;

    // The span is measured between the window's own two events rather than as
    // the difference of two anchor offsets. A second out from the anchor, two
    // `f32` millisecond readings cancel down to some 60 ns of noise — a percent
    // of a small kernel; between the two events themselves the reading is as
    // exact as the device clock.
    let offset = anchor.event.elapsed(start)?;
    let span = start.elapsed(end)?;

    let start_instant = anchor.instant + offset;
    Ok(ProfileTicks::from_start_end(
        start_instant,
        start_instant + span,
    ))
}

/// A point where the device clock and the host clock were read together.
///
/// `cuEventElapsedTime` measures from one event to another and knows nothing of
/// the host clock, so placing a window on the host timeline takes a third event
/// whose host time is known.
#[derive(Debug, Clone)]
struct Anchor {
    /// Shared: a window that has not been read back yet still measures against
    /// the anchor it opened under, which the profiler may have replaced since.
    event: Arc<Event>,
    instant: Instant,
}

impl Anchor {
    /// Record an event on `stream` and wait for it, so the host time taken
    /// right afterwards is the time the device stamped it.
    fn take(stream: CUstream, pool: &EventPool) -> Result<Self, DriverError> {
        let event = pool.acquire()?;
        event.record(stream)?;
        event.wait()?;

        Ok(Self {
            event: Arc::new(event),
            instant: Instant::now(),
        })
    }
}

/// The current [`Anchor`] and the stream it is recorded on.
#[derive(Debug)]
struct Anchoring {
    /// A stream of its own, used for nothing but anchoring. Anchoring waits for
    /// its event, and an anchor recorded into a working stream would wait for
    /// everything queued ahead of it — exactly the drain this profiler exists
    /// to avoid.
    stream: CUstream,
    current: Anchor,
}

// SAFETY: the stream is only ever recorded on and waited for, both of which the
// driver serializes internally, and the profiler owning it is itself only
// reachable under the server's lock.
unsafe impl Send for Anchoring {}

impl Anchoring {
    fn new(pool: &EventPool) -> Result<Self, DriverError> {
        // Non-blocking, so the legacy default stream never synchronizes with it.
        let stream = stream::create(stream::StreamKind::NonBlocking)?;
        let current = Anchor::take(stream, pool)?;

        Ok(Self { stream, current })
    }
}

impl Drop for Anchoring {
    fn drop(&mut self) {
        // SAFETY: `self.stream` was created in `new` and nothing else holds it.
        if let Err(err) = unsafe { stream::destroy(self.stream) } {
            log::warn!("Failed to release the profiling anchor stream: {err}");
        }
    }
}

/// A CUDA event that owns itself: created with timing enabled, destroyed when
/// dropped.
#[derive(Debug)]
struct Event {
    sys: CUevent,
}

// SAFETY: a `CUevent` is a handle into the driver rather than thread-affine
// state. It is recorded on the server's thread and read back on whichever
// thread awaits the profile — the same crossing
// [`Fence`](super::sync::Fence) already makes.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    fn new() -> Result<Self, DriverError> {
        // `CU_EVENT_DEFAULT` rather than the `CU_EVENT_DISABLE_TIMING` a pure
        // synchronization event would take: that flag is exactly the timestamp
        // this profiler reads back.
        let sys = event::create(CUevent_flags::CU_EVENT_DEFAULT)?;

        Ok(Self { sys })
    }

    /// Enqueue the timestamp on `stream`, returning once it is queued.
    fn record(&self, stream: CUstream) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `self`, and `stream` is a
        // stream the caller holds.
        unsafe { event::record(self.sys, stream) }
    }

    /// Block until the device has reached this event.
    fn wait(&self) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `self`.
        unsafe { event::synchronize(self.sys) }
    }

    /// Device time from `self` to `other`. Both must have been reached.
    fn elapsed(&self, other: &Event) -> Result<Duration, DriverError> {
        // SAFETY: both events are live for as long as their references.
        let ms = unsafe { event::elapsed(self.sys, other.sys) }?;

        // Clamped rather than trusted: the device clock is monotonic, but a
        // negative reading would panic the conversion, and a profiler is not
        // worth a panic.
        Ok(Duration::from_secs_f64((ms as f64 / 1000.0).max(0.0)))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `self.sys` was created in `new` and nothing else holds it.
        if let Err(err) = unsafe { event::destroy(self.sys) } {
            log::warn!("Failed to release a profiling event: {err}");
        }
    }
}

/// Recycled profiling events, shared between the profiler and every measurement
/// it has handed out.
///
/// A window's events outlive the profiler's own map: they are read by the
/// future that closed the window, which may be awaited much later and on
/// another thread. So the pool is shared rather than owned, and an event
/// returns to it once the measurement holding it is done.
#[derive(Debug, Clone, Default)]
struct EventPool {
    free: Arc<Mutex<Vec<Event>>>,
}

impl EventPool {
    fn acquire(&self) -> Result<Event, DriverError> {
        match self.free.lock().unwrap().pop() {
            Some(event) => Ok(event),
            None => Event::new(),
        }
    }

    fn release(&self, event: Event) {
        self.free.lock().unwrap().push(event);
    }
}

/// A driver failure as the error the server reports.
fn server_error(error: DriverError) -> ServerError {
    ServerError::Generic {
        reason: format!("{error}"),
        backtrace: BackTrace::capture(),
    }
}

/// A driver failure while setting a window up, as the error a profile reports.
fn profile_error(error: DriverError) -> ProfileError {
    ProfileError::Server(Box::new(server_error(error)))
}
