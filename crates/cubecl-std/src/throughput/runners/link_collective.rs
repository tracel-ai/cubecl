use alloc::{boxed::Box, vec::Vec};
use cubecl_common::profile::{Duration, Instant};
use cubecl_core::{
    future::block_on,
    ir::{ElemType, FloatKind},
};
use cubecl_runtime::{
    client::{Client, ClientGroup, MemberBuffers},
    server::{Collective, Handle, ReduceOperation},
    throughput::ThroughputError,
};

use crate::throughput::{KernelSampler, link_peer::MAX_BYTES, memory_probe};

const REDUCED: ElemType = ElemType::Float(FloatKind::F32);

/// Builds `kind` over every member of `group`, sized as nccl-tests sizes it. `ops_count` is those
/// bytes, which rounds down to a whole `f32` per member.
///
/// # Errors
///
/// [`ThroughputError::Unsupported`] for a collective that is not an all-reduce, an all-gather or a
/// reduce-scatter, a reduction the group cannot do on `f32`, or more than [`MAX_BYTES`];
/// [`ThroughputError::Allocation`] where a member has no room for its buffers; and
/// [`ThroughputError::Launch`] where the collective did not run.
pub fn build_kernel(
    group: &ClientGroup,
    kind: Collective,
    bytes: u64,
) -> Result<KernelSampler, ThroughputError> {
    if bytes > MAX_BYTES {
        return Err(ThroughputError::Unsupported);
    }
    let members = group.clients().len() as u64;
    let unit = REDUCED.size() as u64 * members;
    let bytes = (bytes / unit).max(1) * unit;
    let (source, destination) = match kind {
        Collective::AllReduce => (bytes, bytes),
        Collective::AllGather => (bytes / members, bytes),
        Collective::ReduceScatter => (bytes, bytes / members),
        _ => return Err(ThroughputError::Unsupported),
    };
    if kind != Collective::AllGather && !group.reduces(REDUCED) {
        return Err(ThroughputError::Unsupported);
    }

    // A member that cannot allocate refuses its part, and its peers would wait for it forever.
    let buffers = group
        .clients()
        .iter()
        .map(|client| memory_probe::reserve(client, [source as usize, destination as usize]))
        .collect::<Result<Vec<[Handle; 2]>, _>>()?;

    let written: Vec<(Client, Handle)> = group
        .clients()
        .iter()
        .zip(&buffers)
        .map(|(client, [_, destination])| (client.clone(), destination.clone()))
        .collect();
    let group = group.clone();
    let sample = Box::new(move |iterations: usize| {
        let start = Instant::now();
        // A skip guard held across any call queues nothing for it, and a zero reads as no timing.
        let skipped = (0..iterations).fold(false, |skipped, _| {
            let skips = group.skips_communication();
            queue(&group, kind, &buffers);
            skipped | skips | group.skips_communication()
        });
        for client in group.clients() {
            client.sync_collective();
            let _ = block_on(client.sync());
        }
        match skipped {
            true => Duration::ZERO,
            false => start.elapsed(),
        }
    });

    sample(1);
    for (client, destination) in &written {
        block_on(client.sync_buffers([destination])).map_err(|_| ThroughputError::Launch)?;
    }

    Ok(KernelSampler::new(sample, bytes as usize, 1))
}

fn queue(group: &ClientGroup, kind: Collective, buffers: &[[Handle; 2]]) {
    let buffers = buffers
        .iter()
        .map(|[src, dst]| MemberBuffers {
            src: src.clone(),
            dst: dst.clone(),
        })
        .collect();
    match kind {
        Collective::AllGather => group.all_gather(buffers),
        Collective::ReduceScatter => group.reduce_scatter(buffers, REDUCED, ReduceOperation::Sum),
        Collective::AllReduce => group.all_reduce(buffers, REDUCED, ReduceOperation::Sum),
        other => unreachable!("`build_kernel` refuses {other}"),
    }
}
