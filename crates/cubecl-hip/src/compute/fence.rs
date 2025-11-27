use cubecl_core::server::RuntimeError;
use cubecl_hip_sys::HIP_SUCCESS;

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
        unsafe {
            let status = cubecl_hip_sys::hipEventCreateWithFlags(
                &mut event,
                cubecl_hip_sys::hipEventDefault,
            );
            assert_eq!(status, HIP_SUCCESS, "Should create the stream event");
            let status = cubecl_hip_sys::hipEventRecord(event, stream);
            assert_eq!(status, HIP_SUCCESS, "Should record the stream event");

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
        unsafe {
            let status = cubecl_hip_sys::hipStreamWaitEvent(stream, self.event, 0);
            assert_eq!(
                status, HIP_SUCCESS,
                "Should successfully wait for stream event"
            );
            let status = cubecl_hip_sys::hipEventDestroy(self.event);
            assert_eq!(status, HIP_SUCCESS, "Should destrdestroy the stream eventt");
        }
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](hipStream_t) are completed.
    pub fn wait_sync(self) -> Result<(), RuntimeError> {
        unsafe {
            let status = cubecl_hip_sys::hipEventSynchronize(self.event);

            if status != HIP_SUCCESS {
                return Err(RuntimeError::Generic {
                    context: format!("Should successfully wait for stream event: {status}"),
                });
            }
            let status = cubecl_hip_sys::hipEventDestroy(self.event);

            if status != HIP_SUCCESS {
                return Err(RuntimeError::Generic {
                    context: format!("Should destrdestroy the stream event: {status}"),
                });
            }
        }

        Ok(())
    }
}
