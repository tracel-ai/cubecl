use cudarc::driver::sys::{CUevent_flags, CUevent_st, CUevent_wait_flags, CUstream_st};

/// A fence is simply an [event](CUevent_st) created on a [stream](CUevent_st) that you can wait
/// until completion.
///
/// This is useful for doing synchronization outside of the compute server, which is normally
/// locked by a mutex or a channel. This allows the server to continue accepting other tasks.
#[derive(Debug)]
pub struct Fence {
    event: *mut CUevent_st,
}

// If we don't close the stream or destroy the event, it is safe.
//
// # Safety
//
// Since streams are never closed and we destroy the event after waiting, which consumes the
// [Fence], it is safe.
unsafe impl Send for Fence {}

#[allow(unused)]
impl Fence {
    /// Create a new [Fence] on the given stream.
    ///
    /// # Notes
    ///
    /// The [stream](CUevent_st) must be initialized.
    pub fn new(stream: *mut CUstream_st) -> Self {
        unsafe {
            let event =
                cudarc::driver::result::event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
            cudarc::driver::result::event::record(event, stream).unwrap();

            Self { event }
        }
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](CUstream_st) are completed.
    pub fn wait_sync(self) {
        unsafe {
            cudarc::driver::result::event::synchronize(self.event).unwrap();
            cudarc::driver::result::event::destroy(self.event).unwrap();
        }
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](CUstream_st) are completed on the [original stream](CUstream_st) before new tasks
    /// are registered on the [provided stream](CUstream_st).
    ///
    /// # Notes
    ///
    /// The [stream](CUevent_st) must be initialized.
    pub fn wait_async(self, stream: *mut CUstream_st) {
        unsafe {
            cudarc::driver::result::stream::wait_event(
                stream,
                self.event,
                CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
            .unwrap();
            cudarc::driver::result::event::destroy(self.event).unwrap();
        }
    }
}
