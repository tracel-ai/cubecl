use cubecl_core::{Runtime, client::ComputeClient};
use std::collections::HashMap;

use crate::FarmError;

pub trait FarmClient {
    type R: Runtime;
    type Link;

    fn init(group_split: GroupSplit) -> Result<Farm<Self::R, Self::Link>, FarmError>;
}

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

pub enum GroupSplit {
    Proportional(Vec<f64>),
    Explicit(Vec<usize>),
    SingleGroup,
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

pub fn assign_units_proportional(unit_count: usize, proportions: &[f64]) -> Vec<usize> {
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
