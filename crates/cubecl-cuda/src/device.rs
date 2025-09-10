use cubecl_core::Device;
use cubecl_runtime::id::DeviceId;

// It is not clear if CUDA has a limit on the number of bindings it can hold at
// any given time, but it's highly unlikely that it's more than this. We can
// also assume that we'll never have more than this many bindings in flight,
// so it's 'safe' to store only this many bindings.
pub const CUDA_MAX_BINDINGS: u32 = 1024;

#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct CudaDevice {
    pub index: usize,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            index_id: self.index as u32,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        cudarc::driver::CudaContext::device_count().unwrap_or(0) as usize
    }
}
