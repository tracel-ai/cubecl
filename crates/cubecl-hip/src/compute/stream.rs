use cubecl_common::stream_id::StreamId;
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;

use crate::compute::fence::Fence;

#[derive(Debug)]
pub struct Stream {
    pub(crate) sys: cubecl_hip_sys::hipStream_t,
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
