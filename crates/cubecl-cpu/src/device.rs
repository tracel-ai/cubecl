use cubecl_common::device::{Device, DeviceId, DeviceKind, DeviceRole};

#[derive(new, Clone, PartialEq, Eq, Default, Hash, Debug)]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> DeviceId {
        DeviceId::new(DeviceRole::Runtime, DeviceKind::Cpu, 0)
    }
}
