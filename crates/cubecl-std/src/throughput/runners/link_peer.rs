use alloc::boxed::Box;
use cubecl::prelude::*;
use cubecl_common::profile::Instant;
use cubecl_core as cubecl;
use cubecl_runtime::{
    server::Handle,
    throughput::{KernelConfig, ThroughputError},
};

use crate::throughput::{memory_probe, verify::verify};

/// `u32`s per vector a peer kernel moves: 16 bytes, the widest load every runtime issues.
const VECTOR_WIDTH: usize = 4;

/// The bytes of one vector, the unit every peer transfer is a whole number of.
pub(crate) const VECTOR_BYTES: u64 = 16;

const UNITS_PER_CUBE: u32 = 256;

const MAX_CUBES: u32 = 1024;

/// Builds [`peer_load`] on `destination`, moving `bytes` of `source`'s memory per pass, every load
/// crossing the link. `ops_count` is the bytes moved.
///
/// # Errors
///
/// [`ThroughputError::Allocation`] where either device has no room for its buffer,
/// [`ThroughputError::Unsupported`] where `source` cannot share it, and
/// [`ThroughputError::Launch`] where the kernel did not run.
pub fn build_load(
    source: &Client,
    destination: &Client,
    bytes: u64,
) -> Result<KernelConfig, ThroughputError> {
    let bytes = whole_vectors(bytes);
    let peer = shared(source, bytes)?;
    let [local] = memory_probe::reserve(destination, [bytes as usize])?;

    let launcher = destination.clone();
    let written = local.clone();
    let sample = Box::new(move |iterations: usize| {
        let start = Instant::now();
        for _ in 0..iterations {
            launch_peer_load(&launcher, &peer, &local, bytes);
        }
        // A failure is not this sync's to report: `verify` asked the output.
        let _ = cubecl_core::future::block_on(launcher.sync());
        start.elapsed()
    });
    verify(destination, &sample, &written)?;

    Ok(KernelConfig {
        sample,
        ops_count: bytes as usize,
        min_iterations: 1,
    })
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
    let bytes = whole_vectors(bytes);
    let [local] = memory_probe::reserve(source, [bytes as usize])?;
    let peer = shared(destination, bytes)?;

    let launcher = source.clone();
    let written = peer.clone();
    let sample = Box::new(move |iterations: usize| {
        let start = Instant::now();
        for _ in 0..iterations {
            launch_peer_store(&launcher, &local, &peer, bytes);
        }
        let _ = cubecl_core::future::block_on(launcher.sync());
        start.elapsed()
    });
    verify(destination, &sample, &written)?;

    Ok(KernelConfig {
        sample,
        ops_count: bytes as usize,
        min_iterations: 1,
    })
}

/// A buffer of `bytes` on `owner`, bound for its peers' kernels.
fn shared(owner: &Client, bytes: u64) -> Result<Handle, ThroughputError> {
    let [buffer] = memory_probe::reserve(owner, [bytes as usize])?;
    owner
        .share_with_peers(&buffer)
        .map_err(|_| ThroughputError::Unsupported)
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
