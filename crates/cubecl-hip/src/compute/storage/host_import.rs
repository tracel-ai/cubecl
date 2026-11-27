use crate::compute::status::checked;
use cubecl_environment::bytes::Bytes;
use cubecl_server::driver::DriverError;
use cubecl_server::storage::{HostPinner, HostRegistry};

/// The memory HIP pinned for this process's devices: `hipHostRegister` pins a range once.
static PINNED: HostRegistry = HostRegistry::new();

/// HIP's pinning of host memory, and the events of the stream whose imports it releases.
pub(super) struct HipPinner {
    stream: cubecl_hip_sys::hipStream_t,
}

impl HipPinner {
    /// Orders releases behind `stream`'s work. `stream` must outlive the pinner.
    pub(super) fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self { stream }
    }
}

/// A HIP event, destroyed when dropped.
pub(super) struct HipEvent(cubecl_hip_sys::hipEvent_t);

impl HostPinner for HipPinner {
    type Event = HipEvent;

    fn registry(&self) -> &'static HostRegistry {
        &PINNED
    }

    fn pin(&self, memory: &Bytes) -> Result<(), DriverError> {
        let flags = cubecl_hip_sys::hipHostRegisterPortable | cubecl_hip_sys::hipHostRegisterMapped;
        // SAFETY: the host owns `memory.len()` bytes at its address for as long as an import holds
        // it, which is until `unpin`.
        let status = unsafe {
            cubecl_hip_sys::hipHostRegister(memory.as_ptr() as *mut _, memory.len(), flags)
        };
        checked("hipHostRegister", status)
    }

    fn unpin(&self, memory: &Bytes) -> Result<(), DriverError> {
        // SAFETY: `memory` was registered by `pin`, and is unregistered once.
        let status = unsafe { cubecl_hip_sys::hipHostUnregister(memory.as_ptr() as *mut _) };
        checked("hipHostUnregister", status)
    }

    fn device_ptr(&self, host: *const u8) -> Result<u64, DriverError> {
        let mut ptr: cubecl_hip_sys::hipDeviceptr_t = std::ptr::null_mut();
        // SAFETY: `host` lies in memory registered as mapped, so the driver has a device address
        // for it, written through `ptr`.
        let status =
            unsafe { cubecl_hip_sys::hipHostGetDevicePointer(&mut ptr, host as *mut _, 0) };
        checked("hipHostGetDevicePointer", status)?;
        Ok(ptr as u64)
    }

    fn record(&self) -> Result<HipEvent, DriverError> {
        let mut event: cubecl_hip_sys::hipEvent_t = std::ptr::null_mut();
        // SAFETY: `event` is a valid out-pointer, used only once the status says it was written.
        let status = unsafe {
            cubecl_hip_sys::hipEventCreateWithFlags(
                &mut event,
                cubecl_hip_sys::hipEventDisableTiming,
            )
        };
        checked("hipEventCreateWithFlags", status)?;
        let event = HipEvent(event);
        // SAFETY: the event was just created, and the stream outlives the pinner.
        let status = unsafe { cubecl_hip_sys::hipEventRecord(event.0, self.stream) };
        checked("hipEventRecord", status)?;
        Ok(event)
    }

    fn is_done(&self, event: &HipEvent) -> Result<bool, DriverError> {
        // SAFETY: the event lives until it is dropped.
        let status = unsafe { cubecl_hip_sys::hipEventQuery(event.0) };
        if status == cubecl_hip_sys::hipError_t_hipErrorNotReady {
            return Ok(false);
        }
        checked("hipEventQuery", status)?;
        Ok(true)
    }

    fn synchronize(&self) -> Result<(), DriverError> {
        // SAFETY: no pointer is passed; the call waits for this device's work.
        let status = unsafe { cubecl_hip_sys::hipDeviceSynchronize() };
        checked("hipDeviceSynchronize", status)
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        // SAFETY: the event was created in `record` and is destroyed once, here.
        let status = unsafe { cubecl_hip_sys::hipEventDestroy(self.0) };
        if let Err(err) = checked("hipEventDestroy", status) {
            log::warn!("destroying an import's release event failed: {err}");
        }
    }
}
