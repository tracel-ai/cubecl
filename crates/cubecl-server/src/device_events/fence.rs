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
///
/// For a fence the driver refused to create or record, its stream is synchronized
/// on the spot. Only when that fails too does the fence hold the error.
pub struct EventFence<A: EventApi> {
    /// `None` when the fence was reached at creation.
    event: Result<Option<Event<A>>, ServerError>,
}

impl<A: EventApi> EventFence<A> {
    /// Record a fence at the current position of `stream`.
    ///
    /// When the driver refuses the event, `stream` is synchronized instead, which
    /// keeps every guarantee a fence gives.
    pub fn new(stream: A::Stream) -> Self {
        let event = match Event::new().and_then(|event| event.record(stream).map(|()| event)) {
            Ok(event) => Ok(Some(event)),
            Err(err) => {
                log::warn!(
                    "a fence could not be recorded, synchronizing its stream instead: {err}"
                );
                A::stream_synchronize(stream)
                    .map(|()| None)
                    .map_err(Into::into)
            }
        };

        Self { event }
    }

    /// Block until the device has reached this fence, so everything enqueued on
    /// its stream beforehand is done.
    ///
    /// # Errors
    ///
    /// Returns a [`ServerError`] if the driver refused to create or record the
    /// fence, or if the device was already faulty.
    pub fn wait_sync(self) -> Result<(), ServerError> {
        match self.event? {
            Some(event) => Ok(event.wait()?),
            None => Ok(()),
        }
    }

    /// Make `stream` wait for this fence on the device, so work queued on it
    /// afterwards runs behind the fenced stream's. Does not block the host,
    /// unless the driver refuses the dependency: the host then waits for the
    /// fence itself, which orders the work just the same.
    ///
    /// # Errors
    ///
    /// Returns a [`ServerError`] when the fenced stream's work failed.
    pub fn wait_async(self, stream: A::Stream) -> Result<(), ServerError> {
        let Some(event) = self.event? else {
            return Ok(());
        };
        if let Err(err) = event.wait_async(stream) {
            log::warn!("a stream could not be made to wait on a fence, waiting on the host: {err}");
            event.wait()?;
        }
        Ok(())
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
