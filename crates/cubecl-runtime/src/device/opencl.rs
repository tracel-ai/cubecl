use cubecl_common::device::{Device, DeviceId};

/// A device of the `OpenCL` runtime, named by its index among the devices the runtime can use,
/// GPUs first: the default, 0, is the first GPU when there is one.
#[derive(new, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct OpenClDevice {
    /// The index of the device among the ones the runtime can use.
    pub index: usize,
}

impl core::fmt::Debug for OpenClDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "OpenCl({})", self.index)
    }
}

impl Device for OpenClDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: super::index_id(self.index),
        }
    }
}
