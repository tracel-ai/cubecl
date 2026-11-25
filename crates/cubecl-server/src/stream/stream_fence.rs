use crate::device_events::{Event, EventApi};

/// A point on a stream, where a lazy read was made or just past its copy, which work on another
/// stream waits on rather than on all the work queued there since.
pub trait StreamFence: core::any::Any + Send + Sync + core::fmt::Debug {
    /// Whether the device has finished the work before it.
    fn is_reached(&self) -> bool;
}

impl<A: EventApi> StreamFence for Event<A>
where
    A::Event: Send + Sync,
{
    fn is_reached(&self) -> bool {
        Event::is_reached(self)
    }
}
