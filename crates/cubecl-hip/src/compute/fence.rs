//! Waiting on a stream from outside the server.
//!
//! The server is behind a mutex or a channel, so a synchronize that blocked
//! while holding it would stop every other logical stream too. A [`Fence`] is
//! an event recorded on the stream and handed out: the server records it and
//! returns, and whoever holds it waits on its own time.

use cubecl_core::server::ServerError;
use cubecl_runtime::driver::checked;

/// A fence is simply an [event](hipEvent_t) created on a [stream](hipStream_t) that you can wait
/// until completion.
///
/// This is useful for doing synchronization outside of the compute server, which is normally
/// locked by a mutex or a channel. This allows the server to continue accepting other tasks.
pub struct Fence {
    event: cubecl_hip_sys::hipEvent_t,
}

// If we don't close the stream or destroy the event, it is safe.
//
// # Safety
//
// Since streams are never closed and we destroy the event after waiting, which consumes the
// [Fence], it is safe.
unsafe impl Send for Fence {}

impl Fence {
    /// Create a new [Fence] on the given stream.
    ///
    /// # Notes
    ///
    /// The [stream](hipStream_t) must be initialized.
    pub fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        let mut event: cubecl_hip_sys::hipEvent_t = std::ptr::null_mut();
        // SAFETY: `stream` must be a valid, initialized HIP stream (enforced by the doc
        // contract). The event is created and immediately recorded on the stream. Both
        // operations are asserted to succeed.
        unsafe {
            let status = cubecl_hip_sys::hipEventCreateWithFlags(
                &mut event,
                cubecl_hip_sys::hipEventDefault,
            );
            // Fatal: a fence that never recorded cannot be waited on, and
            // every caller of this takes one by value expecting to be able to.
            checked("hipEventCreateWithFlags", status).expect("the fence needs an event");
            let status = cubecl_hip_sys::hipEventRecord(event, stream);
            checked("hipEventRecord", status).expect("the fence needs its event recorded");

            Self {
                event: event as *mut _,
            }
        }
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](hipStream_t) are completed.
    ///
    /// # Notes
    ///
    /// The [stream](hipStream_t) must be initialized.
    #[allow(unused)]
    pub fn wait_async(self, stream: cubecl_hip_sys::hipStream_t) {
        // SAFETY: `self.event` is a valid event created in `Fence::new`. `stream` must be
        // a valid HIP stream. The event is destroyed after the wait, and `self` is consumed
        // so the event cannot be used again.
        unsafe {
            let status = cubecl_hip_sys::hipStreamWaitEvent(stream, self.event, 0);
            checked("hipStreamWaitEvent", status).expect("the stream has to wait on the fence");
            let status = cubecl_hip_sys::hipEventDestroy(self.event);
            checked("hipEventDestroy", status).expect("the waited-on event has to be released");
        }
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](hipStream_t) are completed.
    pub fn wait_sync(self) -> Result<(), ServerError> {
        // SAFETY: `self.event` is a valid event created in `Fence::new`. We synchronize
        // (block) until the event completes, then destroy it. `self` is consumed so the
        // event cannot be double-freed.
        unsafe {
            let status = cubecl_hip_sys::hipEventSynchronize(self.event);

            checked("hipEventSynchronize", status)?;
            let status = cubecl_hip_sys::hipEventDestroy(self.event);

            checked("hipEventDestroy", status)?;
        }

        Ok(())
    }
}
