use cubecl_common::stream_id::StreamId;
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::stream::StreamBackend;

use crate::compute::fence::Fence;

#[derive(Debug)]
pub struct Stream {
    pub(crate) sys: cubecl_hip_sys::hipStream_t,
    #[allow(unused)] // For debug prints
    pub(crate) id: StreamId,
}

impl Stream {
    pub fn sync(&mut self) {
        unsafe {
            let status = cubecl_hip_sys::hipStreamSynchronize(self.sys);
            assert_eq!(
                status, HIP_SUCCESS,
                "Should successfully synchronize stream"
            );
        };
    }

    pub fn fence(&mut self) -> Fence {
        Fence::new(self.sys)
    }
}

#[derive(Debug)]
pub struct HipStreamBackend;

impl StreamBackend for HipStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(id: StreamId) -> Self::Stream {
        let stream = unsafe {
            let mut stream: cubecl_hip_sys::hipStream_t = std::ptr::null_mut();
            let stream_status = cubecl_hip_sys::hipStreamCreate(&mut stream);
            assert_eq!(stream_status, HIP_SUCCESS, "Should create a stream");
            stream
        };

        Stream { sys: stream, id }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        stream.fence()
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(stream.sys);
    }

    fn wait_event_sync(event: Self::Event) {
        event.wait_sync();
    }
}
