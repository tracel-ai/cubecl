use alloc::boxed::Box;

use cubecl_common::profile::{Duration, Instant, ProfileDuration, ProfileTicks};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::Arc;

use crate::device_events::{Event, EventApi, EventPool, Pooled};
use crate::driver::DriverError;
use crate::server::{ProfileError, ProfilingToken, ServerError};

/// How long an [anchor](Anchor) is trusted before the next window takes a fresh
/// one.
///
/// The device APIs answer an elapsed time in `f32` milliseconds, so the further
/// an event sits from the anchor the coarser its placement on the host clock: a
/// second out, the representable step is about 60 ns; an hour out, a quarter of
/// a millisecond. A window's own duration is measured between its own two
/// events and is unaffected — this only bounds the error on *where* the window
/// lands, which is what a tracing profiler lines its spans up with.
const ANCHOR_MAX_AGE: Duration = Duration::from_secs(1);

/// What a window reports when its events could not be read back.
///
/// The read only fails when the device is already in trouble, and a resolved
/// profile is a duration rather than a result — there is no error to hand back
/// at that point. Reporting zero would give an autotune sweep a candidate that
/// won by failing, so an unreadable window reports a time no real one reaches
/// and says why in the log.
const UNREADABLE: Duration = Duration::from_secs(3600);

/// Device profiling on the GPU's own clock.
///
/// A profiling window is two events recorded into the stream where the window
/// opened and closed. Nothing on the host waits for them: the device stamps
/// each one as the queue reaches it, and the span between the two is read back
/// later, by whoever awaits the measurement.
///
/// That deferral is what makes a measurement mean something when profiles nest.
/// The [system-time profiler](crate::timestamp_profiler::TimestampProfiler) has
/// to drain the stream at both ends of every window to have anything to time,
/// so an inner window's two drains happen *inside* the outer one and are
/// charged to it: the outer measurement grows with the number of inner ones,
/// and every window reads back the launch latency the drain exposed rather than
/// the work. Events are recorded in-queue and cost the host nothing, so an
/// inner window measures its own kernels and the outer one measures the work
/// rather than the profiling.
///
/// The state is the windows currently open, and the anchor that places them on
/// the host clock.
pub struct EventProfiler<A: EventApi> {
    open: HashMap<ProfilingToken, Result<Open<A>, ProfileError>>,
    counter: u64,
    pool: EventPool<A>,
    /// Created with the first window rather than with the profiler: a process
    /// that never profiles should not own a stream and an event for it.
    anchoring: Option<Anchoring<A>>,
}

/// A window that has opened and not yet closed.
struct Open<A: EventApi> {
    start: Pooled<A>,
    /// The anchor this window opened under, kept so that its start and its end
    /// are placed against the same one however many refreshes happen in
    /// between.
    anchor: Anchor<A>,
}

impl<A: EventApi> EventProfiler<A> {
    /// Open a window at the current position of `stream`.
    ///
    /// # Errors
    ///
    /// [`ServerError`] when the device refuses an event or the stream will not
    /// record one. No window is opened and no token is issued, so the caller
    /// owes nothing; the events involved return to the pool.
    pub fn start(&mut self, stream: A::Stream) -> Result<ProfilingToken, ServerError> {
        let anchor = self.anchor()?;
        let start = self.pool.acquire()?;
        start.record(stream)?;

        let token = ProfilingToken { id: self.counter };
        self.counter += 1;
        self.open.insert(token, Ok(Open { start, anchor }));

        Ok(token)
    }

    /// Close the window `token` opened at the current position of `stream`.
    ///
    /// Returns at enqueue time. The duration is read from the device by the
    /// returned future, which is where the wait for the window's work lives.
    ///
    /// # Errors
    ///
    /// [`ProfileError::NotRegistered`] for a token this profiler never issued
    /// or has already closed, the error registered against the window by
    /// [`failure`](Self::failure) when device work failed inside it, and
    /// [`ProfileError::Server`] when the closing event cannot be recorded.
    /// There is nothing to measure in any of those cases and the window is
    /// gone either way.
    pub fn stop(
        &mut self,
        stream: A::Stream,
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

        Ok(ProfileDuration::new_device_time(async move {
            read::<A>(&start, &end, &anchor).unwrap_or_else(|err| {
                log::error!(
                    "Could not read back a {} profiling window ({err}); reporting {UNREADABLE:?} \
                     so nothing mistakes it for a fast one",
                    A::BACKEND
                );
                let now = Instant::now();
                ProfileTicks::from_start_end(now, now + UNREADABLE)
            })
            // `start` and `end` are dropped here, which is what returns them
            // to the pool.
        }))
    }

    /// Drop the window `token` opened without measuring it, for a caller that
    /// has no way to record its end.
    pub fn abandon(&mut self, token: ProfilingToken) {
        self.open.remove(&token);
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

        let error = ProfileError::from(error);
        self.open
            .iter_mut()
            .for_each(|(_, state)| *state = Err(error.clone()));
    }

    /// The anchor a window opening now measures against, refreshed when the
    /// current one has aged past [`ANCHOR_MAX_AGE`].
    fn anchor(&mut self) -> Result<Anchor<A>, DriverError> {
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

impl<A: EventApi> Default for EventProfiler<A> {
    fn default() -> Self {
        Self {
            open: HashMap::default(),
            counter: 0,
            pool: EventPool::default(),
            anchoring: None,
        }
    }
}

impl<A: EventApi> core::fmt::Debug for EventProfiler<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventProfiler")
            .field("backend", &A::BACKEND)
            .field("open", &self.open.len())
            .field("anchored", &self.anchoring.is_some())
            .finish()
    }
}

/// Read a closed window back from the device.
fn read<A: EventApi>(
    start: &Event<A>,
    end: &Event<A>,
    anchor: &Anchor<A>,
) -> Result<ProfileTicks, DriverError> {
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
/// An elapsed-time call measures from one event to another and knows nothing of
/// the host clock, so placing a window on the host timeline takes a third event
/// whose host time is known.
struct Anchor<A: EventApi> {
    /// Shared: a window that has not been read back yet still measures against
    /// the anchor it opened under, which the profiler may have replaced since.
    event: Arc<Pooled<A>>,
    instant: Instant,
}

impl<A: EventApi> Anchor<A> {
    /// Record an event on `stream` and wait for it, so the host time taken
    /// right afterwards is the time the device stamped it.
    fn take(stream: A::Stream, pool: &EventPool<A>) -> Result<Self, DriverError> {
        let event = pool.acquire()?;
        event.record(stream)?;
        event.wait()?;

        Ok(Self {
            event: Arc::new(event),
            instant: Instant::now(),
        })
    }
}

impl<A: EventApi> Clone for Anchor<A> {
    fn clone(&self) -> Self {
        Self {
            event: self.event.clone(),
            instant: self.instant,
        }
    }
}

/// The current [`Anchor`] and the stream it is recorded on.
struct Anchoring<A: EventApi> {
    /// A stream of its own, used for nothing but anchoring. Anchoring waits for
    /// its event, and an anchor recorded into a working stream would wait for
    /// everything queued ahead of it — exactly the drain this profiler exists
    /// to avoid.
    stream: A::Stream,
    current: Anchor<A>,
}

impl<A: EventApi> Anchoring<A> {
    fn new(pool: &EventPool<A>) -> Result<Self, DriverError> {
        let stream = A::stream_create_non_blocking()?;
        let current = Anchor::take(stream, pool)?;

        Ok(Self { stream, current })
    }
}

impl<A: EventApi> Drop for Anchoring<A> {
    fn drop(&mut self) {
        if let Err(err) = A::stream_destroy(self.stream) {
            log::warn!(
                "Failed to release the {} profiling anchor stream: {err}",
                A::BACKEND
            );
        }
    }
}

/// A driver failure while setting a window up, as the error a profile reports.
fn profile_error(error: DriverError) -> ProfileError {
    ProfileError::Server(Box::new(ServerError::from(error)))
}
