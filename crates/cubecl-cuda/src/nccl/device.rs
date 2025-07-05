use crate::CudaDevice;
use cubecl_core::CubeElement;
use cubecl_core::prelude::Numeric;
use std::cell::RefCell;
use std::collections::HashMap;
use cudarc::driver::sys::CUstream;



/// This comes with the CudaDevice. Its added to each device. size is the byte size of the used
/// type and count is the gpu count.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum NcclDevice {
    Linked {
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        count: ::core::ffi::c_int,
        size: usize,
    },
    LinkedRemote {
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        host: String,
        count: ::core::ffi::c_int,
        size: usize,
    },
}

// We assume a f32 in default.
impl Default for NcclDevice {
    fn default() -> Self {
        let id = cudarc::nccl::result::get_uniqueid().unwrap();
        NcclDevice::Linked {
            id,
            group: 0,
            count: 1,
            size: 4,
        }
    }
}

impl NcclDevice {
    fn new_linked<N: CubeElement + Numeric>(
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        count: ::core::ffi::c_int,
    ) -> Self {
        let size = size_of::<N>();
        NcclDevice::Linked { id, group, count, size }
    }

    fn new_linkedremote<N: CubeElement + Numeric> (
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        host: String,
        count: ::core::ffi::c_int,
    ) -> Self {
        let size = size_of::<N>();
        NcclDevice::LinkedRemote {
            id,
            group,
            host,
            count,
            size,
        }
    }

    pub fn id(&self) -> cudarc::nccl::sys::ncclUniqueId {
        match &self {
            NcclDevice::Linked { id, .. } => *id,
            NcclDevice::LinkedRemote { id, .. } => *id,
        }
    }

    pub fn count(&self) -> ::core::ffi::c_int {
        match &self {
            NcclDevice::Linked { count, .. } => *count,
            NcclDevice::LinkedRemote { count, .. } => *count,
        }
    }

    pub fn group(&self) -> usize {
        match &self {
            NcclDevice::Linked { group, .. } => *group,
            NcclDevice::LinkedRemote { group, .. } => *group,
        }
    }

    pub fn size(&self) -> usize {
        match &self {
            NcclDevice::Linked { size, .. } => *size,
            NcclDevice::LinkedRemote { size, .. } => *size,
        }
    }
}

impl CudaDevice {
    /// This function groups every device of the system into one single group of GPUs.
    pub fn single_linked_group<N: CubeElement + Numeric>() -> Vec<Self> {
        let count = cudarc::driver::safe::CudaContext::device_count().unwrap() as usize;
        let mut devices = Vec::new();

        if count > 0 {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();

            for index in 0..count {
                let nccl = NcclDevice::new_linked::<N>(id, 0, count as ::core::ffi::c_int);
                let cuda = CudaDevice {
                    index,
                    nccl: nccl,
                };
                devices.push(cuda);
            }
        }

        devices
    }

    /// Each f64 builds a group in proportion to the sum of the split vector.
    pub fn linked_groups<N: CubeElement + Numeric>(split: Vec<f64>) -> Vec<Self> {
        let total_count = cudarc::driver::safe::CudaContext::device_count().unwrap() as usize;
        let group_sizes = distribute_gpus(split, total_count);

        let mut devices = Vec::new();
        let mut device_index = 0;

        for (group_id, &group_size) in group_sizes.iter().enumerate() {
            if group_size > 0 {
                let id = cudarc::nccl::result::get_uniqueid().unwrap();

                for _ in 0..group_size {
                    let nccl =
                        NcclDevice::new_linked::<N>(id, group_id, group_size as ::core::ffi::c_int);
                    let cuda = CudaDevice {
                        index: device_index,
                        nccl: nccl,
                    };
                    devices.push(cuda);
                    device_index += 1;
                }
            }
        }

        devices
    }


    /// Each f64 builds a group in proportion to the sum of the split vector.
    /// Each String in hosts will be cloned to every device in a specific group.
    pub fn linkedremote_groups<N: CubeElement + Numeric>(split: Vec<f64>, hosts: Vec<String>) -> Vec<Self> {
        let total_count = cudarc::driver::safe::CudaContext::device_count().unwrap() as usize;
        let group_sizes = distribute_gpus(split, total_count);

        let mut devices = Vec::new();
        let mut device_index = 0;

        for (group_id, &group_size) in group_sizes.iter().enumerate() {
            if group_size > 0 {
                let id = cudarc::nccl::result::get_uniqueid().unwrap();
                let host = hosts
                    .get(group_id)
                    .cloned()
                    .unwrap_or_else(|| format!("host_{}", group_id));

                for _ in 0..group_size {
                    let nccl = NcclDevice::new_linkedremote::<N>(
                        id,
                        group_id,
                        host.clone(),
                        group_size as ::core::ffi::c_int,
                    );
                    let cuda = CudaDevice {
                        index: device_index,
                        nccl: nccl,
                    };
                    devices.push(cuda);
                    device_index += 1;
                }
            }
        }

        devices
    }
}

pub fn distribute_gpus(split: Vec<f64>, size: usize) -> Vec<usize> {
    if size < split.len() {
        let mut result = vec![0; split.len()];
        if !result.is_empty() {
            result[0] = size;
        }
        return result;
    }

    let total_proportion: f64 = split.iter().sum();

    if total_proportion == 0.0 {
        return vec![0; split.len()];
    }

    let mut result = Vec::with_capacity(split.len());
    let mut allocated = 0;

    for (i, &proportion) in split.iter().enumerate() {
        if i == split.len() - 1 {
            result.push(size - allocated);
        } else {
            let gpu_count = ((proportion / total_proportion) * size as f64).round() as usize;
            result.push(gpu_count);
            allocated += gpu_count;
        }
    }
    result
}

thread_local! {
    static CUDA_STREAM_CACHE: RefCell<HashMap<CudaDevice, CUstream>> = RefCell::new(HashMap::new());
}

// Bei Client-Erstellung (im selben Thread)
pub fn register_stream(device: CudaDevice, stream: CUstream) {
    CUDA_STREAM_CACHE.with(|cache| {
        cache.borrow_mut().insert(device, stream);
    });
}

// In NCCL-Funktionen (im selben Thread)
pub fn current_stream(device: CudaDevice) -> Option<CUstream> {
    CUDA_STREAM_CACHE.with(|cache| {
        cache.borrow().get(&device).copied()
    })
}


