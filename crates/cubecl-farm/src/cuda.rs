use crate::*;
use cubecl_core::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cudarc::driver::*;
use cudarc::nccl::{self};
use std::collections::HashMap;

impl FarmClient for CudaRuntime {
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
                unit_group_map.entry(group_id).or_default().push(unit_index);
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
                        link,
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
