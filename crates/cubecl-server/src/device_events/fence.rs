use crate::device_events::{Event, EventApi};
#[cfg(multi_threading)]
use crate::memory_management::drop_queue;
use crate::server::ServerError;

/// An event recorded on a stream and handed out, so a caller can wait for that
/// stream's work from outside the server.
///
/// The server sits behind a mutex or a channel, so a synchronize that blocked
/// while holding it would stall every other logical stream too. Recording an
/// event costs the host nothing: the server records one and returns, and
/// whoever holds the fence waits on its own time.
///
/// Named for the trait it is not: `memory_management::drop_queue::Fence` is the
/// contract, and this is the implementation of it that a device event gives
/// you. Backends alias it back to `Fence` for their own call sites. Named in
/// prose rather than linked because that trait is `multi_threading`-only while
/// this type is not — the fence itself is two driver calls and wants no
/// threads.
///
/// Its event is created and destroyed outright rather than recycled through the
/// pool an [`EventProfiler`](super::EventProfiler) keeps: a fence is raised from
/// `StreamBackend::flush`, which is handed a stream and nothing else, so there
/// is no pool in reach without threading one through that trait. Pool these too
/// once something else needs the same argument.
pub struct EventFence<A: EventApi> {
    event: Event<A>,
}

impl<A: EventApi> EventFence<A> {
    /// Record a fence at the current position of `stream`.
    ///
    /// # Panics
    ///
    /// A fence that never recorded cannot be waited on, and every caller takes
    /// one by value expecting to be able to. There is no useful weaker answer
    /// than failing here.
    pub fn new(stream: A::Stream) -> Self {
        let event = Event::new().expect("the fence needs an event");
        event
            .record(stream)
            .expect("the fence needs its event recorded");

        Self { event }
    }

    /// Block until the device has reached this fence, so everything enqueued on
    /// its stream beforehand is done.
    ///
    /// # Errors
    ///
    /// The fault the wait reveals, when the stream itself failed.
    pub fn wait_sync(self) -> Result<(), ServerError> {
        Ok(self.event.wait()?)
    }

    /// Make `stream` wait for this fence on the device, so work queued on it
    /// afterwards runs behind the fenced stream's. Does not block the host.
    ///
    /// # Panics
    ///
    /// A refused dependency would let `stream` run ahead of work it must
    /// follow, which is a wrong answer rather than a slow one.
    pub fn wait_async(self, stream: A::Stream) {
        self.event
            .wait_async(stream)
            .expect("the stream has to wait on the fence");
    }
}

impl<A: EventApi> core::fmt::Debug for EventFence<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}Fence", A::BACKEND)
    }
}

/// The drop queue is where a fence pays off — a freed host buffer waits on one
/// rather than on the server — and it is `multi_threading`-only, so the impl is
/// too. Everything above it is the same fence either way.
#[cfg(multi_threading)]
impl<A: EventApi> drop_queue::Fence for EventFence<A> {
    fn wait(self) -> Result<(), ServerError> {
        self.wait_sync()
    }
}
