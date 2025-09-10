use cubecl_core::Device;
use cubecl_runtime::id::DeviceId;

#[derive(new, Clone, PartialEq, Eq, Default, Hash, Debug)]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}
