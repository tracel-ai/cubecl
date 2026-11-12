use alloc::string::String;
use alloc::vec::Vec;
use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_environment::stream::StreamId;

use crate::server::{Collective, TransportProvider};

/// Names one [`Execution`] among those issued while the process was observed.
///
/// Minted only while an observer is installed, so an unobserved run pays nothing for it. Unique
/// for the process's lifetime, and increasing in the order executions were issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionId(pub(super) u64);

impl ExecutionId {
    pub(crate) fn mint() -> Self {
        use core::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// Something the runtime was asked to run, as an observer is told of it: on the thread that
/// issued it, before it was submitted.
///
/// A record rather than a callback per kind, so an observer keeps what it was told and walks it
/// later: a communication's per-device cost is the walk over its parts filtered by device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Execution {
    /// This execution.
    pub id: ExecutionId,
    /// The execution this one is part of: the communication a part belongs to, or the
    /// [`ExecutionScope`](super::ExecutionScope) open on the issuing thread.
    pub parent: Option<ExecutionId>,
    /// What runs.
    pub kind: ExecutionKind,
}

/// What an [`Execution`] runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionKind {
    /// A kernel, named as it names itself. Pass the name through
    /// [`type_name_format`](crate::config::type_name_format) to shorten it the way the profiling
    /// logger does.
    Kernel {
        /// Where it runs.
        place: ExecutionPlace,
        /// The kernel's name.
        name: &'static str,
    },
    /// A collective or a transfer, as its caller asked for it. It has no place of its own: a
    /// native collective runs once per member, and an emulated one is the transfers and kernels
    /// issued under it, each placed.
    Communication(Communication),
    /// One device's share of its parent communication, on that device's collective or transfer
    /// stream.
    Part {
        /// Where it runs.
        place: ExecutionPlace,
    },
    /// A compute stream reaching the point where it needs what it awaits.
    Wait {
        /// The compute stream that waits.
        place: ExecutionPlace,
        /// What it waits for.
        awaits: WaitTarget,
    },
}

/// Where an execution runs: a device, and which of its streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionPlace {
    /// The device, with the runtime serving it: two runtimes can number their devices alike.
    pub service: ServiceId,
    /// The stream on it.
    pub stream: StreamRole,
}

/// Which stream of a device an execution runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamRole {
    /// A compute stream, the one kernels are launched on.
    Compute(StreamId),
    /// The stream the device's collectives run on.
    Collective,
    /// The stream the device's transfers to and from its peers run on.
    Transfer,
}

/// Data moved between devices, as its caller asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Communication {
    /// A collective over a group of devices.
    Collective {
        /// Which collective.
        operation: Collective,
        /// The group's devices, rank `i` at position `i`.
        members: Vec<DeviceId>,
        /// The size of one member's source: the root's for a broadcast.
        bytes: u64,
        /// How it runs.
        path: CollectivePath,
    },
    /// One buffer copied from one device to another.
    Transfer {
        /// The device holding the source, with its runtime.
        from: ServiceId,
        /// The device receiving it, with its runtime.
        to: ServiceId,
        /// The size of the buffer.
        bytes: u64,
        /// How the bytes travel.
        route: TransferRoute,
    },
}

/// How a collective runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectivePath {
    /// Over the runtime's device transport, one part per member.
    Native(TransportProvider),
    /// Without one: transfers and kernels issued by the caller, each a child of the collective.
    Emulated {
        /// Why the device transport does not run it.
        why: String,
    },
    /// Not at all: a [`CommunicationSkipGuard`](crate::benchmark::CommunicationSkipGuard) holds
    /// one of its devices, so the collective was checked and queued nothing.
    Skipped,
}

/// How a transfer's bytes travel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferRoute {
    /// Device to device, over the runtime's device transport, as a send part and a receive part.
    Device,
    /// Read back to the host and written to the destination. The read blocks the issuing thread,
    /// so it has no part: its cost is host time.
    Host,
}

/// What a [`Wait`](ExecutionKind::Wait) waits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WaitTarget {
    /// One part, on another stream of the same device.
    Part(ExecutionId),
    /// Every collective queued on the device so far.
    QueuedCollectives,
}
