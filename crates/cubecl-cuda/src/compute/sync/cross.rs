use cudarc::driver::sys::{CUevent_flags, CUevent_st, CUevent_wait_flags, CUstream_st};

/// A cross-device [fence](CrossFence), where the waiting is done on a different stream than the stream
/// on which the wait event is produced.
///
/// This is usefull for blocking one stream while waiting for another stream.
pub struct CrossFence {
    consumer_stream: *mut CUstream_st,
    event: *mut CUevent_st,
}

// If we don't close the stream or destroy the event, it is safe.
//
// # Safety
//
// Since streams are never closed and we destroy the event after waiting, which consumes the
// [CrossFence], it is safe.
unsafe impl Send for CrossFence {}

impl CrossFence {
    /// Create a new [CrossFence] on the given stream.
    ///
    /// # Notes
    ///
    /// The [streams](CUevent_st) must be initialized.
    pub fn new(producer_stream: *mut CUstream_st, consumer_stream: *mut CUstream_st) -> Self {
        unsafe {
            let event =
                cudarc::driver::result::event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
            cudarc::driver::result::event::record(event, producer_stream).unwrap();

            Self {
                consumer_stream,
                event,
            }
        }
    }

    /// Wait for the [CrossFence] to be reached on the [consumer stream](CUstream_st),
    /// ensuring that all previous tasks enqueued to the [producer stream](CUstream_st)
    /// are completed.
    ///
    /// # Notes
    ///
    /// The [consumer_stream](CUevent_st) must be initialized.
    pub fn wait(self) {
        unsafe {
            cudarc::driver::result::stream::wait_event(
                self.consumer_stream,
                self.event,
                CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
            .unwrap();
            cudarc::driver::result::event::destroy(self.event).unwrap();
        }
    }
}
