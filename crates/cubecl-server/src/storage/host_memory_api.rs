use super::HostPins;
use crate::device_events::EventApi;
use crate::driver::DriverError;
use cubecl_environment::bytes::Bytes;

/// The driver calls a runtime pins host memory for its devices with. A release is ordered behind
/// its stream's work by the runtime's [`EventApi`].
///
/// # Assumptions
///
/// The current thread's device, or context, is the device a call answers for: the one whose
/// address [`host_device_ptr`](Self::host_device_ptr) gives and whose work
/// [`device_synchronize`](Self::device_synchronize) waits for.
pub trait HostMemoryApi: EventApi {
    /// The memory this runtime pinned, which every device of the runtime shares.
    fn host_pins() -> &'static HostPins;

    /// Pins `memory` and maps it for every device of the process.
    ///
    /// # Errors
    ///
    /// An implementation returns the driver's error when the driver refuses to pin the memory,
    /// such as memory already pinned or past a limit on the memory it pins.
    ///
    /// # Safety
    ///
    /// `memory` stays allocated and in place until [`host_unregister`](Self::host_unregister).
    unsafe fn host_register(memory: &Bytes) -> Result<(), DriverError>;

    /// Undoes [`host_register`](Self::host_register).
    ///
    /// # Errors
    ///
    /// An implementation returns the driver's error when the driver fails to unpin the memory,
    /// which then stays pinned.
    ///
    /// # Safety
    ///
    /// `memory` was registered, and no device of the process reads it any more.
    unsafe fn host_unregister(memory: &Bytes) -> Result<(), DriverError>;

    /// The current device's address of `host`, a byte of pinned memory.
    ///
    /// # Errors
    ///
    /// An implementation returns the driver's error when the driver maps no address of `host`
    /// for the current device.
    fn host_device_ptr(host: *const u8) -> Result<u64, DriverError>;

    /// Waits for every stream of the current device.
    ///
    /// # Errors
    ///
    /// An implementation returns the driver's error when a stream of the device faulted.
    fn device_synchronize() -> Result<(), DriverError>;
}
