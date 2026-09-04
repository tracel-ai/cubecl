use cubecl_common::device::{Device, DeviceId};

/// A device of the HIP runtime, named by its index.
#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct AmdDevice {
    /// The index of the GPU among the ones HIP reports.
    pub index: usize,
}

impl core::fmt::Debug for AmdDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("AmdDevice({})", self.index))
    }
}

impl Device for AmdDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index as u16,
        }
    }
}
