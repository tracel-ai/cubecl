use crate::compute::events::Hip;
use crate::compute::status::checked;
use cubecl_environment::bytes::Bytes;
use cubecl_server::driver::DriverError;
use cubecl_server::storage::{HostMemoryApi, HostPins};

/// The memory HIP pinned for this process's devices: `hipHostRegister` pins a range once.
static PINS: HostPins = HostPins::new();

impl HostMemoryApi for Hip {
    fn host_pins() -> &'static HostPins {
        &PINS
    }

    unsafe fn host_register(memory: &Bytes) -> Result<(), DriverError> {
        let flags = cubecl_hip_sys::hipHostRegisterPortable | cubecl_hip_sys::hipHostRegisterMapped;
        // SAFETY: the caller keeps `memory.len()` bytes at its address until `host_unregister`.
        let status = unsafe {
            cubecl_hip_sys::hipHostRegister(memory.as_ptr() as *mut _, memory.len(), flags)
        };
        checked("hipHostRegister", status)
    }

    unsafe fn host_unregister(memory: &Bytes) -> Result<(), DriverError> {
        // SAFETY: the caller registered `memory`, and no device reads it any more.
        let status = unsafe { cubecl_hip_sys::hipHostUnregister(memory.as_ptr() as *mut _) };
        checked("hipHostUnregister", status)
    }

    fn host_device_ptr(host: *const u8) -> Result<u64, DriverError> {
        let mut ptr: cubecl_hip_sys::hipDeviceptr_t = std::ptr::null_mut();
        // SAFETY: `ptr` is a live out-parameter, and the driver only looks `host` up, failing for
        // memory it has not mapped.
        let status =
            unsafe { cubecl_hip_sys::hipHostGetDevicePointer(&mut ptr, host as *mut _, 0) };
        checked("hipHostGetDevicePointer", status)?;
        Ok(ptr as u64)
    }

    fn device_synchronize() -> Result<(), DriverError> {
        // SAFETY: no pointer is passed; the call waits for the current device's work.
        let status = unsafe { cubecl_hip_sys::hipDeviceSynchronize() };
        checked("hipDeviceSynchronize", status)
    }
}
