use cubecl_common::profile::Duration;

use crate::driver::DriverError;

/// The device event calls an [`EventProfiler`](super::EventProfiler) is built
/// from.
///
/// The C-family device APIs agree on an event model: a timestamp is recorded
/// into a queue, the device stamps it as the queue reaches it, and the span
/// between two of them is read back afterwards. One implementation per backend
/// spells those calls; everything above them — a profiling window, the anchor
/// that places one on the host clock — is written once.
///
/// This is deliberately only what its consumers need. Memory, launches and
/// module loading have their own seams; widening this trait to cover them would
/// make it the device API rather than the device's *event* API, and the name
/// would stop being true.
pub trait EventApi: Send + Sync + 'static {
    /// A timestamp recorded on the device.
    ///
    /// A handle into the driver rather than thread-affine state: it is
    /// recorded on the server's thread and read back on whichever thread waits
    /// for it, so the backend asserts `Send + Sync` on its own newtype, where
    /// the justification is specific enough to check.
    type Event: Send + Sync;

    /// The queue an event is recorded into.
    ///
    /// No `Send` bound: a stream is held by the profiler, which lives inside a
    /// server the backend already asserts is `Send` as a whole. Only the event
    /// crosses threads on its own.
    type Stream: Copy;

    /// The backend as a reader of a log line sees it.
    const BACKEND: &'static str;

    /// Create an event with timing enabled.
    ///
    /// Timing is the point: the flag a pure synchronization event would take
    /// to disable it is exactly the timestamp a profiler reads back.
    ///
    /// # Errors
    ///
    /// [`DriverError`] when the driver refuses the event, which is when the
    /// device is already in trouble or out of resources.
    fn event_create() -> Result<Self::Event, DriverError>;

    /// Release an event. Called from the event's own `Drop`.
    ///
    /// # Errors
    ///
    /// [`DriverError`], which the caller can only log: a destructor has
    /// nowhere to report.
    fn event_destroy(event: &mut Self::Event) -> Result<(), DriverError>;

    /// Enqueue the timestamp on `stream`, returning once it is queued.
    ///
    /// # Errors
    ///
    /// [`DriverError`] when the stream will not take the event — a stream
    /// recording a graph is the case callers screen for in advance.
    fn event_record(event: &Self::Event, stream: Self::Stream) -> Result<(), DriverError>;

    /// Block the calling thread until the device has reached `event`.
    ///
    /// # Errors
    ///
    /// [`DriverError`], which is the fault the wait revealed on the stream the
    /// event was recorded into.
    fn event_wait(event: &Self::Event) -> Result<(), DriverError>;

    /// Device time from `start` to `end`. Both must have been reached.
    ///
    /// Implementations clamp a negative reading to zero rather than trusting
    /// it: the device clock is monotonic, but a negative duration would panic
    /// the conversion, and a profiler is not worth a panic.
    ///
    /// # Errors
    ///
    /// [`DriverError`], including the driver's own "not ready" when either
    /// event has not been reached.
    fn event_elapsed(start: &Self::Event, end: &Self::Event) -> Result<Duration, DriverError>;

    /// Make `stream` wait for `event` on the device, without blocking the host.
    ///
    /// # Errors
    ///
    /// [`DriverError`] when the driver refuses the dependency.
    fn stream_wait_event(stream: Self::Stream, event: &Self::Event) -> Result<(), DriverError>;

    /// Create a stream that does not synchronize with the legacy default one.
    ///
    /// # Errors
    ///
    /// [`DriverError`] when the driver will not create the stream.
    fn stream_create_non_blocking() -> Result<Self::Stream, DriverError>;

    /// Release a stream created by [`stream_create_non_blocking`](Self::stream_create_non_blocking).
    ///
    /// # Errors
    ///
    /// [`DriverError`], which the caller can only log: this runs from a `Drop`.
    fn stream_destroy(stream: Self::Stream) -> Result<(), DriverError>;
}
