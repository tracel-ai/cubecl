use crate::compute::events::driver_error;
use cubecl_environment::bytes::Bytes;
use cubecl_server::driver::DriverError;
use cubecl_server::storage::{HostPinner, HostRegistry};
use cudarc::driver::sys::{self, CUresult};

/// The memory CUDA pinned for this process's devices: `cuMemHostRegister` pins a range once.
static PINNED: HostRegistry = HostRegistry::new();

/// CUDA's pinning of host memory, and the events of the stream whose imports it releases.
pub(super) struct CudaPinner {
    stream: sys::CUstream,
}

impl CudaPinner {
    /// Orders releases behind `stream`'s work. `stream` must outlive the pinner.
    pub(super) fn new(stream: sys::CUstream) -> Self {
        Self { stream }
    }
}

/// A CUDA event, destroyed when dropped.
pub(super) struct CudaHostEvent(sys::CUevent);

fn checked(op: &'static str, status: CUresult) -> Result<(), DriverError> {
    status.result().map_err(|err| driver_error(op, err))
}

impl HostPinner for CudaPinner {
    type Event = CudaHostEvent;

    fn registry(&self) -> &'static HostRegistry {
        &PINNED
    }

    fn pin(&self, memory: &Bytes) -> Result<(), DriverError> {
        let flags = sys::CU_MEMHOSTREGISTER_PORTABLE | sys::CU_MEMHOSTREGISTER_DEVICEMAP;
        // SAFETY: the host owns `memory.len()` bytes at its address for as long as an import holds
        // it, which is until `unpin`.
        let status =
            unsafe { sys::cuMemHostRegister_v2(memory.as_ptr() as *mut _, memory.len(), flags) };
        checked("cuMemHostRegister", status)
    }

    fn unpin(&self, memory: &Bytes) -> Result<(), DriverError> {
        // SAFETY: `memory` was registered by `pin`, and is unregistered once.
        let status = unsafe { sys::cuMemHostUnregister(memory.as_ptr() as *mut _) };
        checked("cuMemHostUnregister", status)
    }

    fn device_ptr(&self, host: *const u8) -> Result<u64, DriverError> {
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: `host` lies in memory registered with `DEVICEMAP`, so the driver has a device
        // address for it, written through `ptr`.
        let status = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut ptr, host as *mut _, 0) };
        checked("cuMemHostGetDevicePointer", status)?;
        Ok(ptr)
    }

    fn record(&self) -> Result<CudaHostEvent, DriverError> {
        let mut event: sys::CUevent = std::ptr::null_mut();
        let flags = sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
        // SAFETY: `event` is a valid out-pointer, used only once the status says it was written.
        let status = unsafe { sys::cuEventCreate(&mut event, flags) };
        checked("cuEventCreate", status)?;
        let event = CudaHostEvent(event);
        // SAFETY: the event was just created, and the stream outlives the pinner.
        let status = unsafe { sys::cuEventRecord(event.0, self.stream) };
        checked("cuEventRecord", status)?;
        Ok(event)
    }

    fn is_done(&self, event: &CudaHostEvent) -> Result<bool, DriverError> {
        // SAFETY: the event lives until it is dropped.
        let status = unsafe { sys::cuEventQuery(event.0) };
        if status == CUresult::CUDA_ERROR_NOT_READY {
            return Ok(false);
        }
        checked("cuEventQuery", status)?;
        Ok(true)
    }

    fn synchronize(&self) -> Result<(), DriverError> {
        // SAFETY: no pointer is passed; the call waits for the current context's work.
        let status = unsafe { sys::cuCtxSynchronize() };
        checked("cuCtxSynchronize", status)
    }
}

impl Drop for CudaHostEvent {
    fn drop(&mut self) {
        // SAFETY: the event was created in `record` and is destroyed once, here.
        let status = unsafe { sys::cuEventDestroy_v2(self.0) };
        if let Err(err) = checked("cuEventDestroy", status) {
            log::warn!("destroying an import's release event failed: {err}");
        }
    }
}
