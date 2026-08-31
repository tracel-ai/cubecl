use alloc::vec::Vec;
use core::ops::Deref;

use cubecl_environment::sync::{Arc, Mutex};

use crate::device_events::EventApi;
use crate::driver::DriverError;

/// A device event that owns itself: created with timing enabled, released when
/// dropped.
pub struct Event<A: EventApi> {
    sys: A::Event,
}

impl<A: EventApi> Event<A> {
    fn new() -> Result<Self, DriverError> {
        Ok(Self {
            sys: A::event_create()?,
        })
    }

    /// Enqueue the timestamp on `stream`, returning once it is queued.
    ///
    /// # Errors
    ///
    /// [`DriverError`] when the stream will not take the event.
    pub fn record(&self, stream: A::Stream) -> Result<(), DriverError> {
        A::event_record(&self.sys, stream)
    }

    /// Block until the device has reached this event.
    ///
    /// # Errors
    ///
    /// [`DriverError`], the fault the wait revealed on the stream.
    pub fn wait(&self) -> Result<(), DriverError> {
        A::event_wait(&self.sys)
    }

    /// Make `stream` wait for this event on the device, without blocking here.
    ///
    /// # Errors
    ///
    /// [`DriverError`] when the driver refuses the dependency.
    pub fn wait_async(&self, stream: A::Stream) -> Result<(), DriverError> {
        A::stream_wait_event(stream, &self.sys)
    }

    /// Device time from `self` to `other`. Both must have been reached.
    ///
    /// # Errors
    ///
    /// [`DriverError`], including "not ready" when either event has not been
    /// reached.
    pub fn elapsed(&self, other: &Self) -> Result<cubecl_common::profile::Duration, DriverError> {
        A::event_elapsed(&self.sys, &other.sys)
    }
}

impl<A: EventApi> Drop for Event<A> {
    fn drop(&mut self) {
        if let Err(err) = A::event_destroy(&mut self.sys) {
            log::warn!("Failed to release a {} event: {err}", A::BACKEND);
        }
    }
}

impl<A: EventApi> core::fmt::Debug for Event<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}Event", A::BACKEND)
    }
}

/// Recycled events, shared between whoever holds the pool and every event it
/// has handed out.
///
/// A profiling window's events outlive the profiler's own map: they are read by
/// the future that closed the window, which may be awaited much later and on
/// another thread. So the pool is shared rather than owned, and an event
/// returns to it when the [`Pooled`] holding it is dropped.
pub(crate) struct EventPool<A: EventApi> {
    free: Arc<Mutex<Vec<Event<A>>>>,
}

impl<A: EventApi> EventPool<A> {
    /// Take an event from the pool, creating one if it is empty.
    ///
    /// The event returns to the pool when the [`Pooled`] is dropped, on every
    /// path — including the ones that never record it.
    ///
    /// # Errors
    ///
    /// [`DriverError`] when the pool was empty and the driver refused a new
    /// event.
    pub(crate) fn acquire(&self) -> Result<Pooled<A>, DriverError> {
        let event = match self.free.lock().pop() {
            Some(event) => event,
            None => Event::new()?,
        };

        Ok(Pooled {
            event: Some(event),
            pool: self.clone(),
        })
    }

    fn release(&self, event: Event<A>) {
        self.free.lock().push(event);
    }
}

impl<A: EventApi> Clone for EventPool<A> {
    fn clone(&self) -> Self {
        Self {
            free: self.free.clone(),
        }
    }
}

impl<A: EventApi> Default for EventPool<A> {
    fn default() -> Self {
        Self {
            free: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl<A: EventApi> core::fmt::Debug for EventPool<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "EventPool<{}>({})", A::BACKEND, self.free.lock().len())
    }
}

/// An [`Event`] borrowed from an [`EventPool`], returned to it when dropped.
///
/// Acquire and release have to happen in pairs, and a window is abandoned,
/// errored out or dropped mid-setup on paths that have no obvious release
/// site. Tying the release to the drop is what removes the step there is to
/// forget.
pub(crate) struct Pooled<A: EventApi> {
    /// `Some` for the whole life of the value; emptied only by `Drop`, which
    /// is how the event moves back into the pool without cloning it.
    event: Option<Event<A>>,
    pool: EventPool<A>,
}

impl<A: EventApi> Deref for Pooled<A> {
    type Target = Event<A>;

    fn deref(&self) -> &Self::Target {
        self.event.as_ref().expect("emptied only by Drop")
    }
}

impl<A: EventApi> Drop for Pooled<A> {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            self.pool.release(event);
        }
    }
}

impl<A: EventApi> core::fmt::Debug for Pooled<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Pooled<{}>", A::BACKEND)
    }
}
