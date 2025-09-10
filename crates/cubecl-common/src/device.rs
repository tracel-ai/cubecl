/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
}

/// Device trait for all cubecl devices.
pub trait Device: Default + Clone + core::fmt::Debug + Send + Sync {
    /// Create a device from its [id](DeviceId).
    fn from_id(device_id: DeviceId) -> Self;
    /// Retrieve the [device id](DeviceId) from the device.
    fn to_id(&self) -> DeviceId;
    /// Returns the number of devices available under the provided type id.
    fn device_count(type_id: u16) -> usize;
    /// Returns the total number of devices that can be handled by the runtime.
    fn device_count_total() -> usize {
        Self::device_count(0)
    }
}
