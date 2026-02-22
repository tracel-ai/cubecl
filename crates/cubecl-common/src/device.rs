use core::cmp::Ordering;

/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
}

/// Device trait for all cubecl devices.
pub trait Device: Default + Clone + core::fmt::Debug + Send + Sync + 'static {
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

impl core::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

impl Ord for DeviceId {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.type_id.cmp(&other.type_id) {
            Ordering::Equal => self.index_id.cmp(&other.index_id),
            other => other,
        }
    }
}

impl PartialOrd for DeviceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Represent a service that runs on a device.
pub trait DeviceService: Send + 'static {
    /// Initializes the service. It is only called once per device
    fn init(device_id: DeviceId) -> Self;
}
