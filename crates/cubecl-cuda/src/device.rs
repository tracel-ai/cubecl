// It is not clear if CUDA has a limit on the number of bindings it can hold at
// any given time, but it's highly unlikely that it's more than this. We can
// also assume that we'll never have more than this many bindings in flight,
// so it's 'safe' to store only this many bindings.
pub const CUDA_MAX_BINDINGS: u32 = 1024;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CudaDevice {
    Device {
        id: usize,
    },
    Remote {
        id: usize,
        host: String,
    },
    Linked {
        id: usize,
        group: usize,
    },
    LinkedRemote {
        id: usize,
        group: usize,
        host: String,
    },
}

pub struct CudaDeviceInfo {
    pub id: usize,
    pub group: usize,
    pub host: Option<String>,
}

impl CudaDeviceInfo {
    fn new(id: usize, group: usize, host: Option<String>) -> Self {
        CudaDeviceInfo {
            id, group, host,
        }
    }
}

impl CudaDevice {
    pub fn id(&self) -> usize {
        match self {
            CudaDevice::Device { id } => *id,
            CudaDevice::Remote { id, .. } => *id,
            CudaDevice::Linked { id, .. } => *id,
            CudaDevice::LinkedRemote { id, .. } => *id,
        }
    }

    pub fn info(&self) -> CudaDeviceInfo {
        match self {
            CudaDevice::Device { id } => CudaDeviceInfo::new(*id, 0, None),
            CudaDevice::Remote { id, host } => CudaDeviceInfo::new(*id, 0, Some(host.clone())),
            CudaDevice::Linked { id, group } => CudaDeviceInfo::new(*id, *group, None),
            CudaDevice::LinkedRemote { id, group, host } => CudaDeviceInfo::new(*id, *group, Some(host.clone())),
        }
    }
}
impl Default for CudaDevice {
    fn default() -> Self {
        CudaDevice::Device { id: 0 }
    }
}
