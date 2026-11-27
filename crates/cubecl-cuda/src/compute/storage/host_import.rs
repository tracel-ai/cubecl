use crate::compute::events::{Cuda, driver_error};
use cubecl_environment::bytes::Bytes;
use cubecl_server::driver::DriverError;
use cubecl_server::storage::{HostMemoryApi, HostPins};
use cudarc::driver::sys::{self, CUresult};

/// The memory CUDA pinned for this process's devices: `cuMemHostRegister` pins a range once.
static PINS: HostPins = HostPins::new();

fn checked(op: &'static str, status: CUresult) -> Result<(), DriverError> {
    status.result().map_err(|err| driver_error(op, err))
}

impl HostMemoryApi for Cuda {
    fn host_pins() -> &'static HostPins {
        &PINS
    }

    unsafe fn host_register(memory: &Bytes) -> Result<(), DriverError> {
        let flags = sys::CU_MEMHOSTREGISTER_PORTABLE | sys::CU_MEMHOSTREGISTER_DEVICEMAP;
        // SAFETY: the caller keeps `memory.len()` bytes at its address until `host_unregister`.
        let status =
            unsafe { sys::cuMemHostRegister_v2(memory.as_ptr() as *mut _, memory.len(), flags) };
        checked("cuMemHostRegister", status)
    }

    unsafe fn host_unregister(memory: &Bytes) -> Result<(), DriverError> {
        // SAFETY: the caller registered `memory`, and no device reads it any more.
        let status = unsafe { sys::cuMemHostUnregister(memory.as_ptr() as *mut _) };
        checked("cuMemHostUnregister", status)
    }

    fn host_device_ptr(host: *const u8) -> Result<u64, DriverError> {
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: `ptr` is a live out-parameter, and the driver only looks `host` up, failing for
        // memory it has not mapped.
        let status = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut ptr, host as *mut _, 0) };
        checked("cuMemHostGetDevicePointer", status)?;
        Ok(ptr)
    }

    fn device_synchronize() -> Result<(), DriverError> {
        // SAFETY: no pointer is passed; the call waits for the current context's work.
        let status = unsafe { sys::cuCtxSynchronize() };
        checked("cuCtxSynchronize", status)
    }
}
