//! CUDA's device events: the driver calls, and the type naming them.
//!
//! What is built on them — a [`Fence`] handed out so a caller can wait on a
//! stream outside the server's lock, and an [`EventProfiler`] timing work on
//! the device's own clock — lives with the shared
//! [`device_events`](cubecl_runtime::device_events) module, along with the
//! design arguments for both.

use cubecl_common::profile::Duration;
use cubecl_runtime::device_events::EventApi;
use cubecl_runtime::driver::DriverError;
use cudarc::driver::result::{event, stream};
use cudarc::driver::sys::{CUevent, CUevent_flags, CUevent_wait_flags, CUstream};

/// A fence, over CUDA's event API.
pub type Fence = cubecl_runtime::device_events::EventFence<Cuda>;

/// The device profiler, over CUDA's event API.
pub type EventProfiler = cubecl_runtime::device_events::EventProfiler<Cuda>;

/// CUDA's event API.
pub struct Cuda;

/// A CUDA event handle.
///
/// The newtype exists for the assertion below; the driver's own lifecycle is
/// the shared [`Event`](cubecl_runtime::device_events::Event)'s job.
pub struct CudaEvent(CUevent);

// SAFETY: a `CUevent` is a handle into the driver rather than thread-affine
// state. It is recorded on the server's thread and waited for on whichever
// thread holds the fence or awaits the profile.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl EventApi for Cuda {
    type Event = CudaEvent;
    type Stream = CUstream;

    const BACKEND: &'static str = "CUDA";

    fn event_create() -> Result<Self::Event, DriverError> {
        // `CU_EVENT_DEFAULT` rather than the `CU_EVENT_DISABLE_TIMING` a pure
        // synchronization event would take: that flag is exactly the timestamp
        // the profiler reads back.
        let sys = named(
            "cuEventCreate",
            event::create(CUevent_flags::CU_EVENT_DEFAULT),
        )?;

        Ok(CudaEvent(sys))
    }

    fn event_destroy(event: &mut Self::Event) -> Result<(), DriverError> {
        // SAFETY: the handle was created in `event_create` and this consumes
        // the only owner of it.
        named("cuEventDestroy", unsafe { event::destroy(event.0) })
    }

    fn event_record(event: &Self::Event, stream: Self::Stream) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `event`, and `stream` is a
        // stream the caller holds.
        named("cuEventRecord", unsafe { event::record(event.0, stream) })
    }

    fn event_wait(event: &Self::Event) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `event`.
        named("cuEventSynchronize", unsafe { event::synchronize(event.0) })
    }

    fn event_elapsed(start: &Self::Event, end: &Self::Event) -> Result<Duration, DriverError> {
        // SAFETY: both events are live for as long as their references.
        let ms = named("cuEventElapsedTime", unsafe {
            event::elapsed(start.0, end.0)
        })?;

        // Clamped rather than trusted: the device clock is monotonic, but a
        // negative reading would panic the conversion, and a profiler is not
        // worth a panic.
        Ok(Duration::from_secs_f64((ms as f64 / 1000.0).max(0.0)))
    }

    fn stream_wait_event(stream: Self::Stream, event: &Self::Event) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `event`, and `stream` is a
        // stream the caller holds.
        named("cuStreamWaitEvent", unsafe {
            stream::wait_event(stream, event.0, CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT)
        })
    }

    fn stream_create_non_blocking() -> Result<Self::Stream, DriverError> {
        named(
            "cuStreamCreate",
            stream::create(stream::StreamKind::NonBlocking),
        )
    }

    fn stream_destroy(stream: Self::Stream) -> Result<(), DriverError> {
        // SAFETY: the stream was created in `stream_create_non_blocking` and
        // nothing else holds it.
        named("cuStreamDestroy", unsafe { stream::destroy(stream) })
    }
}

/// A cudarc result as the runtime's [`DriverError`], named by the entry point.
///
/// cudarc has already decoded the status into its own `CUresult`; this puts the
/// number back with the name that makes it searchable in NVIDIA's headers, and
/// with the `From` impls a caller's `?` expects.
fn named<T>(
    op: &'static str,
    result: Result<T, cudarc::driver::DriverError>,
) -> Result<T, DriverError> {
    result.map_err(|err| DriverError::new(op, err.0 as u32))
}
