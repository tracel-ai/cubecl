//! HIP's half of device profiling: the driver calls, and the type naming them.
//!
//! The design — what a profiling window is, why its events are read back
//! lazily, and how the anchor places one on the host clock — lives with the
//! shared [`EventProfiler`](cubecl_runtime::device_events::EventProfiler).

use cubecl_common::profile::Duration;
use cubecl_hip_sys::{hipEvent_t, hipStream_t};
use cubecl_runtime::device_events::EventApi;
use cubecl_runtime::driver::{DriverError, checked};

/// The device profiler, over HIP's event API.
pub type EventProfiler = cubecl_runtime::device_events::EventProfiler<Hip>;

/// HIP's event API.
pub struct Hip;

/// A HIP event handle.
///
/// The newtype exists for the assertion below; the driver's own lifecycle is
/// the shared [`Event`](cubecl_runtime::device_events::Event)'s job.
pub struct HipEvent(hipEvent_t);

// SAFETY: a `hipEvent_t` is a handle into the driver rather than thread-affine
// state. It is recorded on the server's thread and read back on whichever
// thread awaits the profile — the same crossing [`Fence`](super::fence::Fence)
// already makes.
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl EventApi for Hip {
    type Event = HipEvent;
    type Stream = hipStream_t;

    const BACKEND: &'static str = "HIP";

    fn event_create() -> Result<Self::Event, DriverError> {
        let mut sys: hipEvent_t = core::ptr::null_mut();

        // SAFETY: `sys` is a valid out-pointer, and the status is checked
        // before the handle it writes is used. `hipEventDefault` rather than
        // the `hipEventDisableTiming` a pure synchronization event would take:
        // that flag is exactly the timestamp the profiler reads back.
        let status = unsafe {
            cubecl_hip_sys::hipEventCreateWithFlags(&mut sys, cubecl_hip_sys::hipEventDefault)
        };
        checked("hipEventCreateWithFlags", status)?;

        Ok(HipEvent(sys))
    }

    fn event_destroy(event: &mut Self::Event) -> Result<(), DriverError> {
        // SAFETY: the handle was created in `event_create` and this consumes
        // the only owner of it.
        let status = unsafe { cubecl_hip_sys::hipEventDestroy(event.0) };
        checked("hipEventDestroy", status)
    }

    fn event_record(event: &Self::Event, stream: Self::Stream) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `event`, and `stream` is a
        // stream the caller holds.
        let status = unsafe { cubecl_hip_sys::hipEventRecord(event.0, stream) };
        checked("hipEventRecord", status)
    }

    fn event_wait(event: &Self::Event) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `event`.
        let status = unsafe { cubecl_hip_sys::hipEventSynchronize(event.0) };
        checked("hipEventSynchronize", status)
    }

    fn event_elapsed(start: &Self::Event, end: &Self::Event) -> Result<Duration, DriverError> {
        let mut ms: f32 = 0.0;

        // SAFETY: `ms` is a valid out-pointer and both events are live for as
        // long as their references.
        let status = unsafe { cubecl_hip_sys::hipEventElapsedTime(&mut ms, start.0, end.0) };
        checked("hipEventElapsedTime", status)?;

        // Clamped rather than trusted: the device clock is monotonic, but a
        // negative reading would panic the conversion, and a profiler is not
        // worth a panic.
        Ok(Duration::from_secs_f64((ms as f64 / 1000.0).max(0.0)))
    }

    fn stream_wait_event(stream: Self::Stream, event: &Self::Event) -> Result<(), DriverError> {
        // SAFETY: the event is live for as long as `event`, and `stream` is a
        // stream the caller holds. The flag argument is HIP's only defined
        // one, `hipEventWaitDefault`.
        let status = unsafe { cubecl_hip_sys::hipStreamWaitEvent(stream, event.0, 0) };
        checked("hipStreamWaitEvent", status)
    }

    fn stream_create_non_blocking() -> Result<Self::Stream, DriverError> {
        let mut stream: hipStream_t = core::ptr::null_mut();

        // SAFETY: `stream` is a valid out-pointer, and the status is checked
        // before the handle it writes is used. Non-blocking, so the legacy
        // default stream never synchronizes with it.
        let status = unsafe {
            cubecl_hip_sys::hipStreamCreateWithFlags(
                &mut stream,
                cubecl_hip_sys::hipStreamNonBlocking,
            )
        };
        checked("hipStreamCreateWithFlags", status)?;

        Ok(stream)
    }

    fn stream_destroy(stream: Self::Stream) -> Result<(), DriverError> {
        // SAFETY: the stream was created in `stream_create_non_blocking` and
        // nothing else holds it.
        let status = unsafe { cubecl_hip_sys::hipStreamDestroy(stream) };
        checked("hipStreamDestroy", status)
    }
}
