use crate::reuse::*;

pub trait Link: Sized {
    // Required methods
    fn new(index: usize) -> Self;
    fn new_group(indices: Vec<usize>) -> Vec<Self>;
    // Provided methods
}

pub trait FarmRuntime: Sized {
    type R: Runtime;
    type L: Link;
    type Server: ComputeServer;

    // Required methods
    fn link() -> Self::L;
    fn runtime() -> Self::R;
    fn device(index: usize) -> <Self::R as Runtime>::Device;
    fn device_count() -> usize;

    // Provided methods
    fn farm(split: GroupSplit) -> Result<Farm<Self>> {
        let count = Self::device_count();
        Farm::<Self>::new(0, count, split)
    }
}

pub struct Farm<FR: FarmRuntime> {
    pub id: usize,
    pub unit_count: usize,
    pub groups: Vec<FarmGroup<FR>>,
}
impl<FR: FarmRuntime> Farm<FR> {
    pub fn new(id: usize, unit_count: usize, split: GroupSplit) -> Result<Self> {
        let resolved_split = split.determine_split(unit_count);
        let groups = match resolved_split {
            GroupSplit::SingleGroup => {
                let mut units = Vec::with_capacity(unit_count);
                if unit_count == 1 {
                    let client = <<FR as FarmRuntime>::R as Runtime>::client(
                        &<FR as FarmRuntime>::device(0),
                    );

                    units.push(FarmUnit::Single {
                        device_index: 0,
                        client,
                        handles: None,
                    });
                } else {
                    for i in 0..unit_count {
                        let client = <<FR as FarmRuntime>::R as Runtime>::client(
                            &<FR as FarmRuntime>::device(i),
                        );
                        units.push(FarmUnit::Linked {
                            id: i,
                            device_index: i,
                            client,
                            link: <FR as FarmRuntime>::L::new(i),
                        });
                    }
                }
                vec![FarmGroup::new(0, units)]
            }

            GroupSplit::Explicit(counts) => {
                if counts.iter().sum::<usize>() != unit_count {
                    return Err(FarmError::InvalidSplitConfiguration);
                }
                let mut created_groups = Vec::new();
                let mut device_index_offset = 0;

                for (group_id, &group_size) in counts.iter().enumerate() {
                    if group_size == 0 {
                        continue;
                    }

                    let mut units_for_group = Vec::with_capacity(group_size);

                    if group_size == 1 {
                        let client = <<FR as FarmRuntime>::R as Runtime>::client(
                            &<FR as FarmRuntime>::device(device_index_offset),
                        );
                        units_for_group.push(FarmUnit::Single {
                            device_index: device_index_offset,
                            client,
                            handles: None,
                        });
                    } else {
                        for local_rank in 0..group_size {
                            let device_index = device_index_offset + local_rank;
                            let client = <<FR as FarmRuntime>::R as Runtime>::client(
                                &<FR as FarmRuntime>::device(device_index),
                            );
                            units_for_group.push(FarmUnit::Linked {
                                id: local_rank, // Rank is local to the group
                                device_index,
                                client,
                                link: <FR as FarmRuntime>::L::new(device_index),
                            });
                        }
                    }
                    created_groups.push(FarmGroup::new(group_id, units_for_group));
                    device_index_offset += group_size;
                }
                created_groups
            }
            GroupSplit::Proportional(_) => unreachable!(),
        };
        Ok(Self {
            id,
            unit_count,
            groups,
        })
    }

    pub fn all_units(&self) -> impl Iterator<Item = &FarmUnit<FR>> {
        self.groups.iter().flat_map(|g| g.units.iter())
    }

    pub fn linked_units(&self) -> impl Iterator<Item = &FarmUnit<FR>> {
        self.all_units().filter(|u| u.is_linked())
    }

    pub fn single_units(&self) -> impl Iterator<Item = &FarmUnit<FR>> {
        self.all_units().filter(|u| !u.is_linked())
    }

    pub fn group(&self, group_id: usize) -> Option<&FarmGroup<FR>> {
        self.groups.iter().find(|g| g.id == group_id)
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn total_units(&self) -> usize {
        self.groups.iter().map(|g| g.units.len()).sum()
    }
}

#[derive(Debug)]
pub enum GroupSplit {
    Proportional(Vec<f64>),
    Explicit(Vec<usize>),
    SingleGroup,
}

impl GroupSplit {
    pub fn new_proportional(proportions: Vec<f64>) -> Self {
        GroupSplit::Proportional(proportions)
    }

    pub fn new_explicit(counts: Vec<usize>) -> Self {
        GroupSplit::Explicit(counts)
    }

    pub fn determine_split(&self, device_count: usize) -> Self {
        if device_count == 0 {
            return GroupSplit::Explicit(vec![]);
        }

        if device_count == 1 {
            return GroupSplit::SingleGroup;
        }

        match self {
            GroupSplit::Proportional(proportions) => {
                if proportions.is_empty() || proportions.iter().all(|&p| p <= 0.0) {
                    FarmError::ProportionFallback;
                    return GroupSplit::SingleGroup;
                }
                let assignments = assign_units_proportional(device_count, proportions);
                let mut counts = vec![0; proportions.len()];

                for group_id in assignments {
                    counts[group_id] += 1;
                }
                GroupSplit::Explicit(counts)
            }

            GroupSplit::Explicit(counts) => {
                let sum = counts.iter().sum::<usize>();
                if sum == device_count {
                    GroupSplit::Explicit(counts.clone())
                } else {
                    FarmError::ExplicitFallback { sum, device_count };
                    GroupSplit::SingleGroup
                }
            }

            GroupSplit::SingleGroup => GroupSplit::SingleGroup,
        }
    }
}

pub struct FarmGroup<FR: FarmRuntime> {
    pub id: usize,
    pub units: Vec<FarmUnit<FR>>,
    pub handles: Option<HashMap<usize, Vec<Handle>>>,
}

impl<FR: FarmRuntime> FarmGroup<FR> {
    pub fn new(id: usize, units: Vec<FarmUnit<FR>>) -> Self {
        Self {
            id,
            units,
            handles: None,
        }
    }

    pub fn has_links(&self) -> bool {
        self.units.iter().any(|u| u.is_linked())
    }

    pub fn size(&self) -> usize {
        self.units.len()
    }

    pub fn is_single(&self) -> bool {
        self.units.len() == 1
    }

    pub fn linked_units(&self) -> impl Iterator<Item = &FarmUnit<FR>> {
        self.units.iter().filter(|u| u.is_linked())
    }

    pub fn get_unit(&self, index: usize) -> Result<&FarmUnit<FR>> {
        self.units.get(index).ok_or(FarmError::InvalidDevice)
    }
}

pub enum FarmUnit<FR: FarmRuntime> {
    Linked {
        id: usize,
        device_index: usize,
        client: ComputeClient<
            <<FR as FarmRuntime>::R as Runtime>::Server,
            <<FR as FarmRuntime>::R as Runtime>::Channel,
        >,
        link: <FR as FarmRuntime>::L,
    },
    Single {
        device_index: usize,
        client: ComputeClient<
            <<FR as FarmRuntime>::R as Runtime>::Server,
            <<FR as FarmRuntime>::R as Runtime>::Channel,
        >,
        handles: Option<HashMap<usize, Handle>>,
    },
}

impl<FR: FarmRuntime> FarmUnit<FR> {
    pub fn device_index(&self) -> usize {
        match self {
            FarmUnit::Linked { device_index, .. } => *device_index,
            FarmUnit::Single { device_index, .. } => *device_index,
        }
    }

    pub fn client(
        &self,
    ) -> &ComputeClient<
        <<FR as FarmRuntime>::R as Runtime>::Server,
        <<FR as FarmRuntime>::R as Runtime>::Channel,
    > {
        match self {
            FarmUnit::Linked { client, .. } => client,
            FarmUnit::Single { client, .. } => client,
        }
    }

    pub fn is_linked(&self) -> bool {
        matches!(self, FarmUnit::Linked { .. })
    }

    pub fn link(&self) -> Option<&<FR as FarmRuntime>::L> {
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
    if total == 0.0 {
        return vec![0; unit_count];
    }

    let mut assignments = vec![0; unit_count];
    let mut current_unit = 0;

    for (group_id, &proportion) in proportions.iter().enumerate() {
        let group_size = ((proportion / total) * unit_count as f64).round() as usize;
        let actual_size = group_size.min(unit_count.saturating_sub(current_unit));

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
