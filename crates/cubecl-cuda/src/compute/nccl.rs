use crate::CudaDevice;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum NcclDevice {
    Linked {
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        size: ::core::ffi::c_int,
    },
    LinkedRemote {
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        host: String,
        size: ::core::ffi::c_int,
    },
}

impl Default for NcclDevice {
    fn default() -> Self {
        let id = cudarc::nccl::result::get_uniqueid().unwrap();
        NcclDevice::Linked {
            id,
            group: 0,
            size: 1,
        }
    }
}

impl NcclDevice {
    pub fn new_linked(
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        size: ::core::ffi::c_int,
    ) -> Self {
        NcclDevice::Linked { id, group, size }
    }

    pub fn new_linkedremote(
        id: cudarc::nccl::sys::ncclUniqueId,
        group: usize,
        host: String,
        size: ::core::ffi::c_int,
    ) -> Self {
        NcclDevice::LinkedRemote {
            id,
            group,
            host,
            size,
        }
    }

    pub fn id(&self) -> cudarc::nccl::sys::ncclUniqueId {
        match &self {
            NcclDevice::Linked { id, .. } => *id,
            NcclDevice::LinkedRemote { id, .. } => *id,
        }
    }

    pub fn size(&self) -> ::core::ffi::c_int {
        match &self {
            NcclDevice::Linked { size, .. } => *size,
            NcclDevice::LinkedRemote { size, .. } => *size,
        }
    }

    pub fn group(&self) -> usize {
        match &self {
            NcclDevice::Linked { group, .. } => *group,
            NcclDevice::LinkedRemote { group, .. } => *group,
        }
    }
}

impl CudaDevice {
    fn nccl(&self) -> NcclDevice {
        match &self.nccl {
            None => Default::default(),
            Some(nccl_device) => nccl_device.clone(),
        }
    }

    pub fn single_linked_group() -> Vec<Self> {
        let count = cudarc::driver::safe::CudaContext::device_count().unwrap() as usize;
        let mut devices = Vec::new();

        if count > 0 {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();

            for index in 0..count {
                let nccl = NcclDevice::new_linked(id, 0, count as ::core::ffi::c_int);
                let cuda = CudaDevice {
                    index,
                    nccl: Some(nccl),
                };
                devices.push(cuda);
            }
        }

        devices
    }

    pub fn linked_groups(split: Vec<f64>) -> Vec<Self> {
        let total_count = cudarc::driver::safe::CudaContext::device_count().unwrap() as usize;
        let group_sizes = distribute_gpus(split, total_count);

        let mut devices = Vec::new();
        let mut device_index = 0;

        for (group_id, &group_size) in group_sizes.iter().enumerate() {
            if group_size > 0 {
                let id = cudarc::nccl::result::get_uniqueid().unwrap();

                for _ in 0..group_size {
                    let nccl =
                        NcclDevice::new_linked(id, group_id, group_size as ::core::ffi::c_int);
                    let cuda = CudaDevice {
                        index: device_index,
                        nccl: Some(nccl),
                    };
                    devices.push(cuda);
                    device_index += 1;
                }
            }
        }

        devices
    }

    pub fn linkedremote_groups(split: Vec<f64>, hosts: Vec<String>) -> Vec<Self> {
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
                    let nccl = NcclDevice::new_linkedremote(
                        id,
                        group_id,
                        host.clone(),
                        group_size as ::core::ffi::c_int,
                    );
                    let cuda = CudaDevice {
                        index: device_index,
                        nccl: Some(nccl),
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
