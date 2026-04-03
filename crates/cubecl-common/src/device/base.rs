use crate::stub::Arc;
use core::{any::Any, cmp::Ordering};

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
}

impl core::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "DeviceId(type={}, index={})",
            self.type_id, self.index_id
        ))
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

/// An pointer to a service's server utilities.
pub type ServerUtilitiesHandle = Arc<dyn Any + Send + Sync>;

/// Represent a service that runs on a device.
pub trait DeviceService: Send + 'static {
    /// Initializes the service. It is only called once per device.
    fn init(device_id: DeviceId) -> Self;
    /// Get the service utilities.
    fn utilities(&self) -> ServerUtilitiesHandle;
}
