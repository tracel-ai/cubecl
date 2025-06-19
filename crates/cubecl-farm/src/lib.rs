/*!

Every Farm is composed out of FarmGroup's. In a FarmGroup we find a vector of FarmUnit. When a unit (a single GPU) is linked trough something like a NcclComm it has the state Linked. Otherwise it's seen as Single. With the given implementations we should be able get a good frame to iteraite over our Farm!

For now i will allign the trait FarmCube allong Nccl. The grouping of Nccl here is moved to Rust. This should make the concept more usable for Rust devs. The goal here will be to first just start each device in its own thread to then use a thread per group to manage all concurrent devices. With some helper functions we should be able to send kernels to all devices and so on. Later there will be cpu thread strategies implemented. Also posting to the CPU while having traing loops could be implemented further down the road.

*/

use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cudarc::driver::*;
use cudarc::nccl::{self};
use std::collections::HashMap;

// TODO:! The Iteration is set up. A entry point with proper guards is needed. Something like the ComputeClient just with extra functions, rest will happen in the background on the long run.

pub struct Farm<R: Runtime, L> {
    pub id: usize,
    pub unit_count: usize,
    pub groups: Vec<FarmGroup<R, L>>,
}

impl<R: Runtime, L> Farm<R, L> {
    pub fn all_units(&self) -> impl Iterator<Item = &FarmUnit<R, L>> {
        self.groups.iter().flat_map(|g| g.units.iter())
    }

    pub fn linked_units(&self) -> impl Iterator<Item = &FarmUnit<R, L>> {
        self.all_units().filter(|u| u.is_linked())
    }

    pub fn single_units(&self) -> impl Iterator<Item = &FarmUnit<R, L>> {
        self.all_units().filter(|u| !u.is_linked())
    }

    pub fn group(&self, group_id: usize) -> Option<&FarmGroup<R, L>> {
        self.groups.iter().find(|g| g.id == group_id)
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn total_units(&self) -> usize {
        self.groups.iter().map(|g| g.units.len()).sum()
    }
}

pub struct FarmGroup<R: Runtime, L> {
    pub id: usize,
    pub units: Vec<FarmUnit<R, L>>,
}

impl<R: Runtime, L> FarmGroup<R, L> {
    pub fn has_links(&self) -> bool {
        self.units.iter().any(|u| u.is_linked())
    }

    pub fn size(&self) -> usize {
        self.units.len()
    }

    pub fn is_single(&self) -> bool {
        self.units.len() == 1
    }

    pub fn linked_units(&self) -> impl Iterator<Item = &FarmUnit<R, L>> {
        self.units.iter().filter(|u| u.is_linked())
    }
}

pub enum FarmUnit<R: Runtime, L> {
    Linked {
        id: usize,
        device_index: usize,
        client: ComputeClient<R::Server, R::Channel>,
        link: L,
    },
    Single {
        device_index: usize,
        client: ComputeClient<R::Server, R::Channel>,
    },
}

impl<R: Runtime, L> FarmUnit<R, L> {
    pub fn device_index(&self) -> usize {
        match self {
            FarmUnit::Linked { device_index, .. } => *device_index,
            FarmUnit::Single { device_index, .. } => *device_index,
        }
    }

    pub fn client(&self) -> &ComputeClient<R::Server, R::Channel> {
        match self {
            FarmUnit::Linked { client, .. } => client,
            FarmUnit::Single { client, .. } => client,
        }
    }

    pub fn is_linked(&self) -> bool {
        matches!(self, FarmUnit::Linked { .. })
    }

    pub fn link(&self) -> Option<&L> {
        match self {
            FarmUnit::Linked { link, .. } => Some(link),
            FarmUnit::Single { .. } => None,
        }
    }

    pub fn rank(&self) -> Option<usize> {
        match self {
            FarmUnit::Linked { id, .. } => Some(*id),
            FarmUnit::Single { .. } => None,
        }
    }
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

impl FarmCube for CudaRuntime {
    type R = CudaRuntime;
    type Link = nccl::Comm;

    fn init(group_split: GroupSplit) -> Result<Farm<Self::R, Self::Link>, CudaError> {
        result::init()?;
        let unit_count = result::device::get_count()? as usize;
        let assignments = match (unit_count, group_split) {
            (1, _) => vec![0],
            (n, GroupSplit::SingleGroup) => vec![0; n],
            (n, GroupSplit::Proportional(proportions)) => {
                assign_units_proportional(n, &proportions)
            }
            (_, GroupSplit::Explicit(assignments)) => assignments,
        };

        if assignments.len() > unit_count {
            return Err(CudaError::InvalidConfiguration);
        }
        let mut unit_group_map: HashMap<usize, Vec<usize>> = HashMap::new();

        for (unit_index, &group_id) in assignments.iter().enumerate() {
            if unit_index < unit_count {
                unit_group_map
                    .entry(group_id)
                    .or_insert_with(Vec::new)
                    .push(unit_index);
            }
        }
        let mut groups = Vec::new();

        for (group_id, unit_indices) in unit_group_map {
            let mut units = Vec::new();

            if unit_indices.len() > 1 {
                let nccl_id = nccl::Id::new()?;
                let mut streams = Vec::new();

                for &unit_index in &unit_indices {
                    let ctx = CudaContext::new(unit_index)?;
                    let stream = ctx.default_stream();
                    streams.push(stream);
                }

                for (rank, (&unit_index, stream)) in unit_indices.iter().zip(streams).enumerate() {
                    let link =
                        nccl::Comm::from_rank(stream, rank, unit_indices.len(), nccl_id.clone())?;
                    let unit = CudaDevice { index: unit_index };
                    let client = CudaRuntime::client(&unit);

                    units.push(FarmUnit::Linked {
                        id: rank,
                        device_index: unit_index,
                        client,
                        link: link,
                    });
                }
            } else {
                let unit_index = unit_indices[0];
                let unit = CudaDevice { index: unit_index };
                let client = CudaRuntime::client(&unit);

                units.push(FarmUnit::Single {
                    device_index: unit_index,
                    client,
                });
            }

            groups.push(FarmGroup {
                id: group_id,
                units,
            });
        }
        groups.sort_by_key(|g| g.id);
        Ok(Farm {
            id: 0,
            unit_count,
            groups,
        })
    }
}

fn assign_units_proportional(unit_count: usize, proportions: &[f64]) -> Vec<usize> {
    let total: f64 = proportions.iter().sum();
    let mut assignments = vec![0; unit_count];
    let mut current_unit = 0;

    for (group_id, &proportion) in proportions.iter().enumerate() {
        let group_size = ((proportion / total) * unit_count as f64).round() as usize;
        let actual_size = group_size.min(unit_count - current_unit);

        for _ in 0..actual_size {
            if current_unit < unit_count {
                assignments[current_unit] = group_id;
                current_unit += 1;
            }
        }
    }
    let mut group_id = 0;

    while current_unit < unit_count {
        assignments[current_unit] = group_id % proportions.len();
        current_unit += 1;
        group_id += 1;
    }
    assignments
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
