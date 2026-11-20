use alloc::{boxed::Box, vec, vec::Vec};
use cubecl::prelude::*;
use cubecl_common::profile::Instant;
use cubecl_core as cubecl;
use cubecl_core::ir::{ElemType, UIntKind};
use cubecl_runtime::{
    server::{Handle, PeerAccess},
    throughput::{KernelConfig, ThroughputError},
};

use crate::throughput::{LaunchConfig, memory_probe};

/// `u32`s per vector a peer kernel moves: 16 bytes, the widest load every runtime issues.
const VECTOR_WIDTH: usize = 4;

/// The bytes of one vector, the unit every peer transfer is a whole number of.
pub(crate) const VECTOR_BYTES: u64 = 16;

const UNITS_PER_CUBE: u32 = 256;

const MAX_CUBES: u32 = 1024;

/// The most a link probe moves in one pass: a peer kernel's buffer length in `u32`s is held in 32
/// bits, and every route is held to the same bound so their curves compare.
pub const MAX_BYTES: u64 = u32::MAX as u64 * 4;

/// Builds [`peer_load`] on `destination`, moving `bytes` of `source`'s memory per pass, every load
/// crossing the link. `ops_count` is the bytes moved.
///
/// # Errors
///
/// [`ThroughputError::Unsupported`] where `destination`'s kernels cannot bind `source`'s memory or
/// `bytes` is more than a kernel can address, [`ThroughputError::Allocation`] where either device
/// has no room for its buffer, and [`ThroughputError::Launch`] where the kernel did not run.
pub fn build_load(
    source: &Client,
    destination: &Client,
    bytes: u64,
) -> Result<KernelConfig, ThroughputError> {
    let copy = PeerCopy::load(source, destination, bytes)?;
    build_copies(vec![copy], whole_vectors(bytes), launch_peer_load)
}

/// Builds [`peer_store`] on `source`, moving `bytes` into `destination`'s memory per pass, every
/// store crossing the link. `ops_count` is the bytes moved.
///
/// # Errors
///
/// As [`build_load`].
pub fn build_store(
    source: &Client,
    destination: &Client,
    bytes: u64,
) -> Result<KernelConfig, ThroughputError> {
    let copy = PeerCopy::store(source, destination, bytes)?;
    build_copies(vec![copy], whole_vectors(bytes), launch_peer_store)
}

/// Builds [`peer_store`] on every member at once, each storing `bytes` into the next member's
/// memory and the last into the first's. `ops_count` is the bytes one member moves.
///
/// # Errors
///
/// As [`build_load`], for any member and the next, and [`ThroughputError::Unsupported`] for fewer
/// than two members or a device named twice.
pub fn build_ring_store(members: &[Client], bytes: u64) -> Result<KernelConfig, ThroughputError> {
    let copies = ring(members)
        .ok_or(ThroughputError::Unsupported)?
        .map(|(source, destination)| PeerCopy::store(source, destination, bytes))
        .collect::<Result<Vec<_>, _>>()?;
    build_copies(copies, whole_vectors(bytes), launch_peer_store)
}

/// Builds [`peer_load`] on every member at once, each loading `bytes` from the previous member's
/// memory and the first from the last's. `ops_count` is the bytes one member moves.
///
/// # Errors
///
/// As [`build_ring_store`].
pub fn build_ring_load(members: &[Client], bytes: u64) -> Result<KernelConfig, ThroughputError> {
    let copies = ring(members)
        .ok_or(ThroughputError::Unsupported)?
        .map(|(source, destination)| PeerCopy::load(source, destination, bytes))
        .collect::<Result<Vec<_>, _>>()?;
    build_copies(copies, whole_vectors(bytes), launch_peer_load)
}

/// The buffers of one peer kernel: `launcher` runs it, reading `read` and writing `written`, which
/// `owner` holds.
struct PeerCopy {
    launcher: Client,
    owner: Client,
    read: Handle,
    written: Handle,
}

impl PeerCopy {
    /// [`peer_load`]'s: `destination` loads `bytes` of `source`'s memory into its own.
    fn load(source: &Client, destination: &Client, bytes: u64) -> Result<Self, ThroughputError> {
        let bytes = movable(destination, source, bytes)?;
        let peer = initialized(source, bytes)?;
        let peer = source
            .share_with_peers(&peer)
            .map_err(|_| ThroughputError::Unsupported)?;
        let [local] = memory_probe::reserve(destination, [bytes as usize])?;
        Ok(Self {
            launcher: destination.clone(),
            owner: destination.clone(),
            read: peer,
            written: local,
        })
    }

    /// [`peer_store`]'s: `source` stores `bytes` of its own memory into `destination`'s.
    fn store(source: &Client, destination: &Client, bytes: u64) -> Result<Self, ThroughputError> {
        let bytes = movable(source, destination, bytes)?;
        let local = initialized(source, bytes)?;
        let [peer] = memory_probe::reserve(destination, [bytes as usize])?;
        let peer = destination
            .share_with_peers(&peer)
            .map_err(|_| ThroughputError::Unsupported)?;
        Ok(Self {
            launcher: source.clone(),
            owner: destination.clone(),
            read: local,
            written: peer,
        })
    }
}

impl core::fmt::Debug for PeerCopy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PeerCopy")
            .field("read", &self.read)
            .field("written", &self.written)
            .finish_non_exhaustive()
    }
}

/// Whether `members` form a ring whose every member's kernels can bind the next member's memory.
pub(crate) fn ring_stores(members: &[Client]) -> bool {
    ring(members).is_some_and(|mut hops| {
        hops.all(|(source, destination)| source.peer_access(destination) >= PeerAccess::Memory)
    })
}

/// Whether `members` form a ring whose every member's kernels can bind the previous member's
/// memory.
pub(crate) fn ring_loads(members: &[Client]) -> bool {
    ring(members).is_some_and(|mut hops| {
        hops.all(|(source, destination)| destination.peer_access(source) >= PeerAccess::Memory)
    })
}

/// Each member paired with the next and the last with the first, or `None` where the members are
/// fewer than two or name one device twice.
fn ring(members: &[Client]) -> Option<impl Iterator<Item = (&Client, &Client)>> {
    let distinct = members.iter().enumerate().all(|(index, member)| {
        members[..index]
            .iter()
            .all(|other| other.service_id() != member.service_id())
    });
    (members.len() >= 2 && distinct).then(|| members.iter().zip(members.iter().cycle().skip(1)))
}

/// Times every copy launched at once, `bytes` each. `ops_count` is one copy's bytes.
fn build_copies(
    copies: Vec<PeerCopy>,
    bytes: u64,
    launch: fn(&Client, &Handle, &Handle, u64),
) -> Result<KernelConfig, ThroughputError> {
    let written: Vec<(Client, Handle)> = copies
        .iter()
        .map(|copy| (copy.owner.clone(), copy.written.clone()))
        .collect();
    let sample = Box::new(move |iterations: usize| {
        let start = Instant::now();
        for _ in 0..iterations {
            for copy in &copies {
                launch(&copy.launcher, &copy.read, &copy.written, bytes);
            }
        }
        // A failure is not this sync's to report: every output was asked before timing.
        for copy in &copies {
            let _ = cubecl_core::future::block_on(copy.launcher.sync());
        }
        start.elapsed()
    });

    // `verify`'s check, on every copy's output.
    sample(1);
    for (owner, written) in &written {
        cubecl_core::future::block_on(owner.sync_buffers([written]))
            .map_err(|_| ThroughputError::Launch)?;
    }

    Ok(KernelConfig {
        sample,
        ops_count: bytes as usize,
        min_iterations: 1,
    })
}

/// `bytes` in whole vectors, where `kernel`'s kernels can bind `owner`'s memory and address them.
fn movable(kernel: &Client, owner: &Client, bytes: u64) -> Result<u64, ThroughputError> {
    if kernel.peer_access(owner) < PeerAccess::Memory || bytes > MAX_BYTES {
        return Err(ThroughputError::Unsupported);
    }
    Ok(whole_vectors(bytes))
}

/// A buffer on `owner` holding at least `bytes`, every line of it written, since a kernel's input
/// must be.
fn initialized(owner: &Client, bytes: u64) -> Result<Handle, ThroughputError> {
    let dtype = ElemType::UInt(UIntKind::U32);
    let config = LaunchConfig::for_device(owner, dtype);
    let line_bytes = (config.vector_size * dtype.size()) as u64;
    let lines = bytes.div_ceil(line_bytes);
    let [buffer] = memory_probe::reserve(owner, [(lines * line_bytes) as usize])?;
    memory_probe::prime(owner, &buffer, lines as usize, config, dtype);
    Ok(buffer)
}

/// `bytes` rounded down to whole vectors, at least one.
pub(crate) fn whole_vectors(bytes: u64) -> u64 {
    (bytes / VECTOR_BYTES).max(1) * VECTOR_BYTES
}

/// Queues [`peer_load`] on `destination`, copying the first `bytes` of `peer`, another device's
/// shared buffer, into `local`. `bytes` is whole vectors, and both buffers hold at least as many.
pub(crate) fn launch_peer_load(destination: &Client, peer: &Handle, local: &Handle, bytes: u64) {
    peer_load::launch(
        destination,
        cube_count(bytes),
        CubeDim::new_1d(UNITS_PER_CUBE),
        VECTOR_WIDTH,
        peer_arg(peer, bytes),
        buffer_arg(local, bytes),
    );
}

/// Queues [`peer_store`] on `source`, copying the first `bytes` of `local` into `peer`, another
/// device's shared buffer. `bytes` is whole vectors, and both buffers hold at least as many.
pub(crate) fn launch_peer_store(source: &Client, local: &Handle, peer: &Handle, bytes: u64) {
    peer_store::launch(
        source,
        cube_count(bytes),
        CubeDim::new_1d(UNITS_PER_CUBE),
        VECTOR_WIDTH,
        buffer_arg(local, bytes),
        peer_arg(peer, bytes),
    );
}
/// Copies `peer` into `local`, every load crossing the link.
///
/// Reads `local.len()` vectors from the peer's memory and writes as many to its own.
#[cube(launch)]
fn peer_load<N: Size>(peer: &Peer<[Vector<u32, N>]>, local: &mut [Vector<u32, N>]) {
    let stride = CUBE_DIM as usize * CUBE_COUNT;
    let mut index = ABSOLUTE_POS;
    while index < local.len() {
        local[index] = peer[index];
        index += stride;
    }
}

/// Copies `local` into `peer`, every store crossing the link, and ends once the stores are visible
/// to the peer.
///
/// Reads `local.len()` vectors from its own memory and writes as many to the peer's.
#[cube(launch)]
fn peer_store<N: Size>(local: &[Vector<u32, N>], peer: &mut Peer<[Vector<u32, N>]>) {
    let stride = CUBE_DIM as usize * CUBE_COUNT;
    let mut index = ABSOLUTE_POS;
    while index < local.len() {
        peer[index] = local[index];
        index += stride;
    }
    // One fence per cube: a fence per unit costs more than its stores below a few MiB.
    sync_cube();
    if UNIT_POS == 0 {
        fence(MemoryOrder::Release, SyncScope::System);
    }
}

fn cube_count(bytes: u64) -> CubeCount {
    let vectors = bytes / VECTOR_BYTES;
    let cubes = (vectors.div_ceil(UNITS_PER_CUBE as u64) as u32).clamp(1, MAX_CUBES);
    CubeCount::Static(cubes, 1, 1)
}

fn peer_arg(handle: &Handle, bytes: u64) -> PeerArg {
    // SAFETY: the caller's handle holds at least `bytes`.
    unsafe { PeerArg::from_raw_parts(handle, (bytes / 4) as usize) }
}

fn buffer_arg(handle: &Handle, bytes: u64) -> BufferArg {
    // SAFETY: the caller's handle holds at least `bytes`.
    unsafe { BufferArg::from_raw_parts(handle.clone(), (bytes / 4) as usize) }
}
