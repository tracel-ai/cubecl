use std::{collections::HashMap, sync::Arc};

use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_cuda::{CudaDevice, CudaRuntime, RuntimeOptions};
use cubecl_runtime::id::DeviceId;
use cudarc::driver::*;
use cudarc::nccl::{self, Comm};

struct Farm<R: Runtime, C, I> {
    id: usize,
    gpu_count: usize,
    clients: Arc<HashMap<I, ComputeClient<R::Server, R::Channel>>>,
    comms: Option<Vec<Option<C>>>,
}

pub enum GroupSplit {
    Propotional(Vec<f64>),
    Explicit(Vec<usize>),
    SingleGroup,
    SingleGpu,
}

pub trait CubeFarm {
    type R: Runtime;
    type Comm;
    type Device;

    fn init(
        id: usize,
        group: GroupSplit,
    ) -> Result<Farm<Self::R, Self::Comm, Self::Device>, CudaError>;
}

impl CubeFarm for CudaRuntime {
    type R = CudaRuntime;
    type Comm = nccl::Comm;
    type Device = CudaDevice;

    fn init(
        id: usize,
        group: GroupSplit,
    ) -> Result<Farm<Self::R, Self::Comm, Self::Device>, CudaError> {
        result::init()?;
        let gpu_count = result::device::get_count()? as usize;

        let mut clients_map = HashMap::new();
        let mut devices = Vec::new();
        let mut contexts = Vec::new();

        match (gpu_count, group) {
            (1, _) => {
                let device_id = CudaDevice { index: 0, group: 0 };
                let client = CudaRuntime::client(&device_id);
                let ctx = CudaContext::new(0)?;
                contexts.push(ctx);
                clients_map.insert(device_id.clone(), client);
                devices.push(device_id);
            }

            (n, GroupSplit::SingleGroup) if n > 1 => {
                for gpu_index in 0..n {
                    let device_id = CudaDevice {
                        index: gpu_index,
                        group: 0,
                    };
                    let client = CudaRuntime::client(&device_id);
                    let ctx = CudaContext::new(gpu_index)?;
                    contexts.push(ctx);
                    clients_map.insert(device_id.clone(), client);
                    devices.push(device_id);
                }
            }

            (n, GroupSplit::SingleGpu) if n > 1 => {
                if id < n {
                    let device_id = CudaDevice {
                        index: id,
                        group: id,
                    };
                    let client = CudaRuntime::client(&device_id);
                    let ctx = CudaContext::new(id)?;
                    contexts.push(ctx);
                    clients_map.insert(device_id.clone(), client);
                    devices.push(device_id);
                } else {
                    return Err(CudaError::InvalidDevice);
                }
            }

            (n, GroupSplit::Propotional(proportions)) if n > 1 => {
                let gpu_assignments = gpus_proportional(n, &proportions);
                for (group_id, gpu_indices) in gpu_assignments.iter().enumerate() {
                    if group_id == id {
                        for &gpu_index in gpu_indices {
                            let device_id = CudaDevice {
                                index: gpu_index,
                                group: group_id,
                            };
                            let client = CudaRuntime::client(&device_id);
                            let ctx = CudaContext::new(gpu_index)?;
                            contexts.push(ctx);
                            clients_map.insert(device_id.clone(), client);
                            devices.push(device_id);
                        }
                    }
                }
            }

            (n, GroupSplit::Explicit(assignments)) if n > 1 => {
                for (gpu_index, &group_id) in assignments.iter().enumerate() {
                    if gpu_index < n && group_id == id {
                        let device_id = CudaDevice {
                            index: gpu_index,
                            group: group_id,
                        };
                        let client = CudaRuntime::client(&device_id);
                        let ctx = CudaContext::new(gpu_index)?;
                        contexts.push(ctx);
                        clients_map.insert(device_id.clone(), client);
                        devices.push(device_id);
                    }
                }
            }

            _ => {
                return Err(CudaError::InvalidConfiguration);
            }
        }

        let comms = if devices.len() > 1 {
            let mut devices_by_group: HashMap<usize, Vec<&CudaDevice>> = HashMap::new();
            for device in &devices {
                devices_by_group
                    .entry(device.group)
                    .or_insert_with(Vec::new)
                    .push(device);
            }
            let mut all_comms = Vec::new();
            for (group_id, group_devices) in devices_by_group {
                if group_devices.len() > 1 {
                    let nccl_id = nccl::Id::new()?;
                    let mut group_streams = Vec::new();
                    let mut group_device_indices = Vec::new();
                    for device in group_devices {
                        let ctx = CudaContext::new(device.index)?;
                        let stream = ctx.default_stream();
                        group_streams.push(stream);
                        group_device_indices.push(device.index);
                    }
                    for (rank, (stream, device_index)) in group_streams
                        .into_iter()
                        .zip(group_device_indices.iter())
                        .enumerate()
                    {
                        let comm = nccl::Comm::from_rank(
                            stream,
                            rank,
                            group_device_indices.len(),
                            nccl_id.clone(),
                        )?;
                        if devices
                            .iter()
                            .any(|d| d.index == *device_index && d.group == group_id)
                        {
                            all_comms.push(Some(comm));
                        }
                    }
                } else {
                    break;
                }
            }

            if all_comms.is_empty() {
                None
            } else {
                Some(all_comms)
            }
        } else {
            None
        };

        Ok(Farm {
            id,
            gpu_count,
            clients: Arc::new(clients_map),
            comms,
        })
    }
}

fn gpus_proportional(gpu_count: usize, proportions: &[f64]) -> Vec<Vec<usize>> {
    let total: f64 = proportions.iter().sum();
    let mut groups = vec![vec![]; proportions.len()];
    let mut assigned = 0;
    for (i, &prop) in proportions.iter().enumerate() {
        let count = ((prop / total) * gpu_count as f64).round() as usize;
        let actual_count = count.min(gpu_count - assigned);

        for j in 0..actual_count {
            groups[i].push(assigned + j);
        }
        assigned += actual_count;
    }
    let mut group_idx = 0;
    let group_count = groups.len();
    while assigned < gpu_count {
        groups[group_idx % group_count].push(assigned);
        assigned += 1;
        group_idx += 1;
    }

    groups
}

#[derive(Debug)]
enum CudaError {
    DriverError,
    InvalidDevice,
    InvalidConfiguration,
    NcclError,
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        CudaError::DriverError
    }
}

impl From<nccl::result::NcclError> for CudaError {
    fn from(err: nccl::result::NcclError) -> Self {
        CudaError::NcclError
    }
}
