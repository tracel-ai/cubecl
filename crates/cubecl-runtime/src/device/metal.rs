use cubecl_common::device::{Device, DeviceId};

/// A device of the native Metal runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum MetalDevice {
    /// The device Metal picks by default, usually the first GPU.
    #[default]
    DefaultDevice,
    /// The discrete GPU at this index among the machine's discrete GPUs.
    DiscreteGpu(usize),
    /// The integrated GPU at this index among the machine's integrated GPUs.
    IntegratedGpu(usize),
    /// Existing device with unique ID, registered through the Metal runtime.
    Existing(u32),
}

impl Device for MetalDevice {
    /// # Panics
    ///
    /// When `device_id`'s type is not one a [`MetalDevice`] writes.
    fn from_id(device_id: DeviceId) -> Self {
        match device_id.type_id {
            0 => Self::DefaultDevice,
            1 => Self::DiscreteGpu(device_id.index_id as usize),
            2 => Self::IntegratedGpu(device_id.index_id as usize),
            3 => Self::Existing(device_id.index_id as u32),
            _ => panic!("Invalid Metal device ID: {:?}", device_id),
        }
    }

    fn to_id(&self) -> DeviceId {
        match self {
            Self::DefaultDevice => DeviceId {
                type_id: 0,
                index_id: 0,
            },
            Self::DiscreteGpu(idx) => DeviceId {
                type_id: 1,
                index_id: super::index::index_id(*idx),
            },
            Self::IntegratedGpu(idx) => DeviceId {
                type_id: 2,
                index_id: super::index::index_id(*idx),
            },
            Self::Existing(id) => DeviceId {
                type_id: 3,
                index_id: *id as u16,
            },
        }
    }
}

impl core::fmt::Display for MetalDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DefaultDevice => write!(f, "Metal (default)"),
            Self::DiscreteGpu(idx) => write!(f, "Metal DiscreteGpu {}", idx),
            Self::IntegratedGpu(idx) => write!(f, "Metal IntegratedGpu {}", idx),
            Self::Existing(id) => write!(f, "Metal Device {}", id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_round_trips_its_id() {
        for device in [
            MetalDevice::DefaultDevice,
            MetalDevice::DiscreteGpu(1),
            MetalDevice::IntegratedGpu(2),
            MetalDevice::Existing(3),
        ] {
            assert_eq!(MetalDevice::from_id(device.to_id()), device);
        }
    }

    #[test]
    #[should_panic(expected = "Invalid Metal device ID")]
    fn an_id_of_an_unknown_type_is_refused() {
        MetalDevice::from_id(DeviceId::new(9, 0));
    }

    #[test]
    fn display_names_the_kind_and_index() {
        assert_eq!(
            alloc::string::ToString::to_string(&MetalDevice::DiscreteGpu(1)),
            "Metal DiscreteGpu 1"
        );
    }
}
