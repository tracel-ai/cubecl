use crate::compute::sync::Fence;
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::stream::StreamBackend;

#[derive(Debug)]
pub struct Stream {
    pub sys: cudarc::driver::sys::CUstream,
    #[allow(unused)] // For debug prints
    pub(crate) id: StreamId,
}

impl Stream {
    pub fn fence(&mut self) -> Fence {
        Fence::new(self.sys)
    }
}

#[derive(Debug)]
pub struct CudaStreamBackend;

impl StreamBackend for CudaStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn init_stream(id: StreamId) -> Self::Stream {
        let stream = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .expect("Can create a new stream.");

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
