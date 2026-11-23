use cubecl_common::device::{Device, DeviceId};

/// A device of the CUDA runtime, named by its index.
#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct CudaDevice {
    /// The index of the GPU among the ones CUDA reports.
    pub index: usize,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Cuda({})", self.index)
    }
}

impl Device for CudaDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: super::index::index_id(self.index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_round_trips_its_id() {
        let device = CudaDevice::new(3);

        assert_eq!(CudaDevice::from_id(device.to_id()), device);
    }

    #[test]
    fn an_index_too_wide_for_the_id_is_kept_at_the_largest() {
        let too_wide = CudaDevice::new(usize::MAX);

        assert_eq!(
            CudaDevice::from_id(too_wide.to_id()).index,
            u16::MAX as usize
        );
    }
}
