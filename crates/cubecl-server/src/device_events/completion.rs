//! A completion boundary over a backend's event API.
//!
//! CUDA and HIP both express "the work submitted so far" as an event recorded
//! on the stream, queried without blocking or waited on to block. What differs
//! between them is only the driver call, which [`EventApi`] already names, so
//! the completion lives here rather than once per backend.

use super::{Event, EventApi};
use crate::server::{BackendCompletion, ServerError};

/// The completion of the work a stream had submitted when the event was
/// recorded.
///
/// The event is created without timing and released with the last handle, after
/// it has been recorded — which is safe on both CUDA and HIP.
pub struct EventCompletion<A: EventApi> {
    event: Event<A>,
}

impl<A: EventApi> EventCompletion<A> {
    /// Record a boundary at the work `stream` has submitted so far.
    ///
    /// # Errors
    ///
    /// The driver's refusal to create or record the event.
    pub fn record(stream: A::Stream) -> Result<Self, ServerError> {
        let event = Event::new_sync()?;
        event.record(stream)?;
        Ok(Self { event })
    }
}

impl<A: EventApi> BackendCompletion for EventCompletion<A> {
    fn is_complete(&self) -> Result<bool, ServerError> {
        Ok(self.event.is_complete()?)
    }

    fn wait(&self) -> Result<(), ServerError> {
        Ok(self.event.wait()?)
    }
}

impl<A: EventApi> core::fmt::Debug for EventCompletion<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}Completion", A::BACKEND)
    }
}
