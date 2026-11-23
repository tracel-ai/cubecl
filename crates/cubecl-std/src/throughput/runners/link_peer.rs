use alloc::{boxed::Box, vec::Vec};
use core::cell::Cell;
use cubecl::prelude::*;
use cubecl_common::profile::{Duration, Instant};
use cubecl_core as cubecl;
use cubecl_core::ir::{ElemType, UIntKind};
use cubecl_runtime::{
    server::{Handle, PeerAccess},
    throughput::ThroughputError,
};

use crate::throughput::{KernelConfig, LaunchConfig, memory_probe};

/// `u32`s per vector a peer kernel moves: 16 bytes, the widest load every runtime issues.
const VECTOR_WIDTH: usize = 4;

/// The bytes of one vector, the unit every peer transfer is a whole number of.
pub(crate) const VECTOR_BYTES: u64 = 16;

const UNITS_PER_CUBE: u32 = 256;

const MAX_CUBES: u32 = 1024;

/// The most a link probe moves in one pass: a peer kernel's buffer length in `u32`s is held in 32
/// bits, and every route is held to the same bound so their curves compare.
pub const MAX_BYTES: u64 = u32::MAX as u64 * 4;

/// Which side of a link a peer copy's kernel runs on.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PeerCopyKind {
    /// [`peer_load`] on the destination, every load crossing the link.
    Load,
    /// [`peer_store`] on the source, every store crossing the link.
    Store,
}

impl PeerCopyKind {
    /// The side whose kernel runs.
    fn launcher<'a>(self, source: &'a Client, destination: &'a Client) -> &'a Client {
        match self {
            Self::Load => destination,
            Self::Store => source,
        }
    }

    /// Whether the side this kind launches on can bind the other side's memory.
    pub(crate) fn reaches(self, source: &Client, destination: &Client) -> bool {
        match self {
            Self::Load => destination.peer_access(source) >= PeerAccess::Memory,
            Self::Store => source.peer_access(destination) >= PeerAccess::Memory,
        }
    }
}

/// Times peer copies launched together: one between a pair, or one from each member of a ring to
/// the next. `ops_count` is the bytes one copy moves.
pub(crate) struct PeerCopier {
    copies: Vec<PeerCopy>,
    kind: PeerCopyKind,
    bytes: u64,
    /// Set once a launch read its timing. Warmup doubles its count on each zero sample, so a later
    /// sample would run up to a thousand passes.
    interrupted: Cell<bool>,
}

/// The buffers of one peer kernel: `launcher` runs it, reading `read` and writing `written`, which
/// `owner` holds.
struct PeerCopy {
    pub launcher: Client,
    pub owner: Client,
    pub read: Handle,
    pub written: Handle,
}

impl PeerCopier {
    /// One copy of `bytes` from `source` to `destination`, refused where the side `kind` launches
    /// on cannot bind the other's memory, launches there read their timing, or `bytes` is more than
    /// a kernel can address.
    pub(crate) fn pair(
        source: &Client,
        destination: &Client,
        kind: PeerCopyKind,
        bytes: u64,
    ) -> Result<Self, ThroughputError> {
        Self::new([(source, destination)].into_iter(), kind, bytes)
    }

    /// One copy of `bytes` from every member to the next, the last to the first, refused as
    /// [`pair`](Self::pair) for any of them, and for fewer than two members or a device named twice.
    pub(crate) fn ring(
        members: &[Client],
        kind: PeerCopyKind,
        bytes: u64,
    ) -> Result<Self, ThroughputError> {
        let hops = Self::hops(members).ok_or(ThroughputError::Unsupported)?;
        Self::new(hops, kind, bytes)
    }

    /// Whether [`ring`](Self::ring) accepts `members`, without allocating.
    pub(crate) fn supports_ring(members: &[Client], kind: PeerCopyKind) -> bool {
        Self::hops(members).is_some_and(|mut hops| {
            hops.all(|(source, destination)| kind.reaches(source, destination))
        })
    }

    /// Runs one pass, confirms every copy's output was written, then hands the copies over to be
    /// timed.
    pub(crate) fn into_config(self) -> Result<KernelConfig, ThroughputError> {
        // `verify`'s check, on every copy's output, which a pass stopped early never wrote.
        if self.time_iterations(1).is_zero() {
            return Err(ThroughputError::Unsupported);
        }
        for copy in &self.copies {
            cubecl_core::future::block_on(copy.owner.sync_buffers([&copy.written]))
                .map_err(|_| ThroughputError::Launch)?;
        }
        let ops_count = self.bytes as usize;
        Ok(KernelConfig {
            sample: Box::new(move |iterations| self.time_iterations(iterations)),
            ops_count,
            min_iterations: 1,
        })
    }

    fn new<'a>(
        hops: impl Iterator<Item = (&'a Client, &'a Client)>,
        kind: PeerCopyKind,
        bytes: u64,
    ) -> Result<Self, ThroughputError> {
        if bytes > MAX_BYTES {
            return Err(ThroughputError::Unsupported);
        }
        let bytes = (bytes / VECTOR_BYTES).max(1) * VECTOR_BYTES;
        let copies = hops
            .map(|(source, destination)| Self::copy(source, destination, kind, bytes))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            copies,
            kind,
            bytes,
            interrupted: Cell::new(false),
        })
    }

    /// Each member paired with the next and the last with the first, or `None` where the members
    /// are fewer than two or name one device twice.
    pub(crate) fn hops(members: &[Client]) -> Option<impl Iterator<Item = (&Client, &Client)>> {
        let distinct = members.iter().enumerate().all(|(index, member)| {
            members[..index]
                .iter()
                .all(|other| other.service_id() != member.service_id())
        });
        (members.len() >= 2 && distinct).then(|| members.iter().zip(members.iter().cycle().skip(1)))
    }

    /// The buffers of one copy of `bytes`, whole vectors, from `source`'s memory to
    /// `destination`'s.
    fn copy(
        source: &Client,
        destination: &Client,
        kind: PeerCopyKind,
        bytes: u64,
    ) -> Result<PeerCopy, ThroughputError> {
        if !kind.reaches(source, destination) {
            return Err(ThroughputError::Unsupported);
        }
        // A launch that reads its timing ends before the next is issued, so none would overlap.
        if kind.launcher(source, destination).launches_read_timing() {
            return Err(ThroughputError::Unsupported);
        }
        let initialized = Self::initialized(source, bytes)?;
        let [empty] = memory_probe::reserve(destination, [bytes as usize])?;
        match kind {
            PeerCopyKind::Load => Ok(PeerCopy {
                launcher: destination.clone(),
                owner: destination.clone(),
                read: Self::shared(source, &initialized)?,
                written: empty,
            }),
            PeerCopyKind::Store => Ok(PeerCopy {
                launcher: source.clone(),
                owner: destination.clone(),
                read: initialized,
                written: Self::shared(destination, &empty)?,
            }),
        }
    }

    /// A buffer on `owner` holding at least `bytes`, every line of it written, since a kernel's
    /// input must be.
    fn initialized(owner: &Client, bytes: u64) -> Result<Handle, ThroughputError> {
        let dtype = ElemType::UInt(UIntKind::U32);
        let config = LaunchConfig::for_device(owner, dtype);
        let line_bytes = (config.vector_size * dtype.size()) as u64;
        let lines = bytes.div_ceil(line_bytes);
        let [buffer] = memory_probe::reserve(owner, [(lines * line_bytes) as usize])?;
        memory_probe::prime(owner, &buffer, lines as usize, config, dtype);
        Ok(buffer)
    }

    fn shared(owner: &Client, buffer: &Handle) -> Result<Handle, ThroughputError> {
        owner
            .share_with_peers(buffer)
            .map_err(|_| ThroughputError::Unsupported)
    }

    fn time_iterations(&self, iterations: usize) -> Duration {
        if self.interrupted.get() {
            return Duration::ZERO;
        }
        let start = Instant::now();
        // Stops at a launch that read its timing: the sample is lost, and more would only cost.
        let waited = (0..iterations).flat_map(|_| &self.copies).any(|copy| {
            self.launch(copy);
            copy.launcher.launches_read_timing()
        });
        // A failure is not this sync's to report: every output was asked before timing.
        for copy in &self.copies {
            let _ = cubecl_core::future::block_on(copy.launcher.sync());
        }
        self.interrupted.set(waited);
        // An observation that began after `copy` checked; a zero reads as no timing.
        match waited {
            true => Duration::ZERO,
            false => start.elapsed(),
        }
    }

    fn launch(&self, copy: &PeerCopy) {
        match self.kind {
            PeerCopyKind::Load => {
                launch_peer_load(&copy.launcher, &copy.read, &copy.written, self.bytes)
            }
            PeerCopyKind::Store => {
                launch_peer_store(&copy.launcher, &copy.read, &copy.written, self.bytes)
            }
        }
    }
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
