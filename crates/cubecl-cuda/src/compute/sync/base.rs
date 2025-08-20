use cudarc::driver::sys::{CUevent_flags, CUevent_st, CUevent_wait_flags, CUstream_st};

/// A stream synchronization point that blocks until all previously enqueued work in the stream
/// has completed.
///
/// Unlike [`Fence`], which creates an event to track a specific point in the stream's execution,
/// `StreamSync` synchronizes the entire stream when [`wait`](StreamSync::wait) is called. This is
/// equivalent to calling `cudaStreamSynchronize` in CUDA.
///
/// # Notes
///
/// - This provides a simpler but potentially less efficient synchronization mechanism compared to
///   [`Fence`], as it waits for all previous operations rather than a specific point.
/// - The stream must remain valid until [`wait`](StreamSync::wait) is called.
/// - This operation is relatively expensive as it blocks the CPU until all GPU operations complete.
pub struct SyncStream {
    stream: *mut CUstream_st,
}

// Safety: Since streams are never closed and synchronization is handled through
// CUDA's stream synchronization API, it is safe to send across threads.
unsafe impl Send for SyncStream {}

impl SyncStream {
    /// Creates a new [`SyncStream`] for the given CUDA stream.
    pub fn new(stream: *mut CUstream_st) -> Self {
        Self { stream }
    }

    /// Blocks the current thread until all previously enqueued work in the stream has completed.
    ///
    /// This operation synchronizes the entire stream, ensuring that all GPU operations
    /// previously enqueued into this stream have completed before returning.
    pub fn wait(self) {
        unsafe {
            cudarc::driver::result::stream::synchronize(self.stream).unwrap();
        }
    }
}
