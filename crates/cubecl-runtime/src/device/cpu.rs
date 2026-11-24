use cubecl_common::device::{Device, DeviceId};

/// The device of the CPU runtime. There is only one.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_id_names_the_one_cpu_device() {
        assert_eq!(CpuDevice::from_id(DeviceId::new(7, 7)), CpuDevice);
    }

    #[test]
    fn the_cpu_device_round_trips_its_id() {
        assert_eq!(CpuDevice::from_id(CpuDevice.to_id()), CpuDevice);
    }
}
