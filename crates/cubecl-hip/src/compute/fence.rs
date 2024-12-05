use cubecl_hip_sys::HIP_SUCCESS;

/// A fence is simply an [event](hipEvent_t) created on a [stream](hipStream_t) that you can wait
/// until completion.
///
/// This is useful for doing synchronization outside of the compute server, which is normally
/// locked by a mutex or a channel. This allows the server to continue accepting other tasks.
pub struct Fence {
    stream: cubecl_hip_sys::hipStream_t,
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
                stream,
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
    pub fn wait(self) {
        unsafe {
            let status = cubecl_hip_sys::hipStreamWaitEvent(self.stream, self.event, 0);
            assert_eq!(
                status, HIP_SUCCESS,
                "Should successfully wait for stream event"
            );
            let status = cubecl_hip_sys::hipEventDestroy(self.event);
            assert_eq!(status, HIP_SUCCESS, "Should destrdestroy the stream eventt");
        }
    }
}

/// A stream synchronization point that blocks until all previously enqueued work in the stream
/// has completed.
///
/// Unlike [`Fence`], which creates an event to track a specific point in the stream's execution,
/// `StreamSync` synchronizes the entire stream when [`wait`](StreamSync::wait) is called. This is
/// equivalent to calling `hipStreamSynchronize` in HIP.
///
/// # Notes
///
/// - This provides a simpler but potentially less efficient synchronization mechanism compared to
///   [`Fence`], as it waits for all previous operations rather than a specific point.
/// - The stream must remain valid until [`wait`](StreamSync::wait) is called.
/// - This operation is relatively expensive as it blocks the CPU until all GPU operations complete.
pub struct SyncStream {
    stream: cubecl_hip_sys::hipStream_t,
}

// Safety: Since streams are never closed and synchronization is handled through
// HIP stream synchronization API, it is safe to send across threads.
unsafe impl Send for SyncStream {}

impl SyncStream {
    /// Creates a new [`SyncStream`] for the given HIP stream.
    pub fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self { stream }
    }

    /// Blocks the current thread until all previously enqueued work in the stream has completed.
    ///
    /// This operation synchronizes the entire stream, ensuring that all GPU operations
    /// previously enqueued into this stream have completed before returning.
    pub fn wait(self) {
        unsafe {
            let status = cubecl_hip_sys::hipStreamSynchronize(self.stream);
            assert_eq!(
                status, HIP_SUCCESS,
                "Should successfully wait for fence event in stream"
            )
        }
    }
}
