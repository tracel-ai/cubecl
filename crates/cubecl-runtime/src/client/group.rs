//! Collectives queued by one caller over the clients of several devices.

use super::{Client, PEER_ORDER};
use crate::server::{
    Agreement, Collective, CollectiveBuffer, ComputeWait, Handle, Overlap, ReduceOperation,
    Reduction, Server, Vote,
};
use alloc::{collections::BTreeSet, vec::Vec};
use cubecl_common::{device::DeviceId, device_handle::DeviceHandle};
use cubecl_ir::ElemType;
use thiserror::Error;

/// The clients of distinct devices of one runtime that run collectives together, for a caller
/// holding every member's buffers.
///
/// Member `i` is rank `i`, in ascending device order. Work queued after a collective reads its
/// result, but per-device `all_reduce` calls are not ordered against it, and on a device both use
/// the two can hang. Buffers move as bytes, so a pitched handle's row padding counts toward its
/// chunks.
#[derive(Clone)]
pub struct ClientGroup {
    clients: Vec<Client>,
    device_ids: Vec<DeviceId>,
    reduces: BTreeSet<ElemType>,
}

/// One member's buffers in a collective.
#[derive(Debug)]
pub struct MemberBuffers {
    /// What the member contributes.
    pub src: Handle,
    /// Where the member's result lands.
    pub dst: Handle,
}

/// Why clients cannot form a [`ClientGroup`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ClientGroupError {
    /// There are no clients.
    #[error("a client group needs at least one client")]
    Empty,
    /// The device handle blocks the caller, which collectives cannot run on.
    #[error("collectives cannot run on a blocking device handle")]
    BlockingDeviceHandle,
    /// The clients run different runtimes, which share no transport.
    #[error("the clients run different runtimes")]
    MixedRuntimes,
    /// The runtime has no transport between its devices.
    #[error("{runtime} has no transport between its devices")]
    NoDeviceTransport {
        /// The runtime's name.
        runtime: &'static str,
    },
    /// The devices are not in strictly ascending order, so a member would not be its rank, or a
    /// device would hold two ranks.
    #[error("the devices are not in strictly ascending order: {devices:?}")]
    NotAscending {
        /// The devices, in the order given.
        devices: Vec<DeviceId>,
    },
}

/// Whether a collective may run with a member's source and destination on the same bytes.
#[derive(Debug, Clone, Copy)]
enum InPlace {
    Allowed,
    Refused,
}

impl ClientGroup {
    /// A group of `clients`, one per member.
    ///
    /// # Errors
    ///
    /// When there are none, when the device handle blocks, when they mix runtimes, when their
    /// devices are not in strictly ascending order, and when their runtime has no device
    /// transport.
    pub fn new(clients: Vec<Client>) -> Result<Self, ClientGroupError> {
        let first = clients.first().ok_or(ClientGroupError::Empty)?;
        if DeviceHandle::<dyn Server>::is_blocking() {
            return Err(ClientGroupError::BlockingDeviceHandle);
        }
        let runtime = first.service_id().service;
        if clients
            .iter()
            .any(|client| client.service_id().service != runtime)
        {
            return Err(ClientGroupError::MixedRuntimes);
        }
        let device_ids: Vec<DeviceId> = clients
            .iter()
            .map(|client| client.device.device_id())
            .collect();
        if device_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ClientGroupError::NotAscending {
                devices: device_ids,
            });
        }
        let Some(transports) = clients
            .iter()
            .map(|client| client.utilities.device_transport.as_ref())
            .collect::<Option<Vec<_>>>()
        else {
            return Err(ClientGroupError::NoDeviceTransport {
                runtime: first.name(),
            });
        };
        // NCCL refuses a reduction unless every member's device supports it.
        let mut reduces = transports[0].reduces.clone();
        for transport in &transports[1..] {
            reduces.retain(|dtype| transport.reduces.contains(dtype));
        }
        Ok(Self {
            clients,
            device_ids,
            reduces,
        })
    }

    /// The members' clients, member `i` first being rank `i`.
    pub fn clients(&self) -> &[Client] {
        &self.clients
    }

    /// Whether the members reduce elements of `dtype` together. Every other collective moves bytes,
    /// whatever their type.
    pub fn reduces(&self, dtype: ElemType) -> bool {
        self.reduces.contains(&dtype)
    }

    /// Every member's `dst` receives every member's `src` reduced by `op`. A member's `src` may be
    /// its `dst`.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, buffers))
    )]
    pub fn all_reduce(&self, buffers: Vec<MemberBuffers>, dtype: ElemType, op: ReduceOperation) {
        let collective = Collective::AllReduce;
        self.expect_reduces(collective, dtype);
        let src = self.expect_buffers(collective, &buffers, InPlace::Allowed);
        Client::expect_whole_elements(collective, dtype, src);
        Self::expect_sizes(
            collective,
            CollectiveBuffer::Destination,
            buffers.iter().map(|it| &it.dst),
            src,
        );
        self.queue(collective, buffers, |_, client, buffers, device_ids, vote| {
            client.queue_all_reduce(
                buffers.src,
                buffers.dst,
                Reduction { dtype, op },
                device_ids,
                ComputeWait::Now,
                vote,
            )
        });
    }

    /// Every member's `dst` receives every member's `src`, concatenated in member order.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, buffers))
    )]
    pub fn all_gather(&self, buffers: Vec<MemberBuffers>) {
        let collective = Collective::AllGather;
        let src = self.expect_buffers(collective, &buffers, InPlace::Refused);
        let members = self.clients.len() as u64;
        Self::expect_sizes(
            collective,
            CollectiveBuffer::Destination,
            buffers.iter().map(|it| &it.dst),
            src * members,
        );
        self.queue(collective, buffers, |_, client, buffers, device_ids, vote| {
            client.queue_all_gather(buffers.src, buffers.dst, device_ids, vote)
        });
    }

    /// Member `i`'s `dst` receives chunk `i` of every member's `src` reduced by `op`.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, buffers))
    )]
    pub fn reduce_scatter(
        &self,
        buffers: Vec<MemberBuffers>,
        dtype: ElemType,
        op: ReduceOperation,
    ) {
        let collective = Collective::ReduceScatter;
        self.expect_reduces(collective, dtype);
        let src = self.expect_buffers(collective, &buffers, InPlace::Refused);
        let chunk = self.expect_chunks(collective, src);
        Client::expect_whole_elements(collective, dtype, chunk);
        Self::expect_sizes(
            collective,
            CollectiveBuffer::Destination,
            buffers.iter().map(|it| &it.dst),
            chunk,
        );
        self.queue(collective, buffers, |_, client, buffers, device_ids, vote| {
            let reduction = Reduction { dtype, op };
            client.queue_reduce_scatter(buffers.src, buffers.dst, reduction, device_ids, vote)
        });
    }

    /// Every member's `dst` receives `src`, sent by the member whose device holds it. That
    /// member's `dst` may be `src`.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src, dst))
    )]
    pub fn broadcast(&self, src: Handle, dst: Vec<Handle>) {
        let collective = Collective::Broadcast;
        self.expect_parts(collective, dst.len());
        let root = self
            .clients
            .iter()
            .position(|client| client.service_id() == src.service)
            .unwrap_or_else(|| {
                panic!(
                    "a broadcast's source is on {}, which is no member's device",
                    src.service
                )
            });
        self.expect_member_buffers(collective, root, &src, &dst[root], InPlace::Allowed);
        for (member, dst) in dst.iter().enumerate() {
            self.expect_on_member(member, dst);
        }
        Self::expect_sizes(
            collective,
            CollectiveBuffer::Destination,
            dst.iter(),
            src.size_in_used(),
        );

        let root_id = self.device_ids[root];
        self.queue(collective, dst, |member, client, dst, device_ids, vote| {
            let src = (member == root).then(|| src.clone());
            client.queue_broadcast(src, dst, root_id, device_ids, vote)
        });
    }

    /// Member `i`'s `dst` receives chunk `i` of every member's `src`, concatenated in member
    /// order.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, buffers))
    )]
    pub fn all_to_all(&self, buffers: Vec<MemberBuffers>) {
        let collective = Collective::AllToAll;
        let src = self.expect_buffers(collective, &buffers, InPlace::Refused);
        self.expect_chunks(collective, src);
        Self::expect_sizes(
            collective,
            CollectiveBuffer::Destination,
            buffers.iter().map(|it| &it.dst),
            src,
        );
        self.queue(collective, buffers, |_, client, buffers, device_ids, vote| {
            client.queue_all_to_all(buffers.src, buffers.dst, device_ids, vote)
        });
    }

    /// Checks one part per member, each member's buffers, and sources of one size; returns the
    /// size of each source.
    fn expect_buffers(
        &self,
        collective: Collective,
        buffers: &[MemberBuffers],
        in_place: InPlace,
    ) -> u64 {
        self.expect_parts(collective, buffers.len());
        for (member, buffers) in buffers.iter().enumerate() {
            self.expect_member_buffers(collective, member, &buffers.src, &buffers.dst, in_place);
        }
        let src = buffers[0].src.size_in_used();
        Self::expect_sizes(
            collective,
            CollectiveBuffer::Source,
            buffers.iter().map(|it| &it.src),
            src,
        );
        src
    }

    fn expect_parts(&self, collective: Collective, parts: usize) {
        let members = self.clients.len();
        assert!(
            parts == members,
            "{collective} over {members} members takes one part each, not {parts}"
        );
    }

    fn expect_reduces(&self, collective: Collective, dtype: ElemType) {
        assert!(
            self.reduces(dtype),
            "{collective} on {} cannot reduce {dtype:?}",
            self.clients[0].name()
        );
    }

    /// Checks `size` bytes split evenly over the members, and returns the size of a chunk.
    fn expect_chunks(&self, collective: Collective, size: u64) -> u64 {
        let members = self.clients.len() as u64;
        assert!(
            size.is_multiple_of(members),
            "{collective} over {members} members cannot split {size} bytes evenly"
        );
        size / members
    }

    /// Checks a member's source and destination are on its device, and apart unless the
    /// collective runs in place on exactly the same bytes.
    fn expect_member_buffers(
        &self,
        collective: Collective,
        member: usize,
        src: &Handle,
        dst: &Handle,
        in_place: InPlace,
    ) {
        self.expect_on_member(member, src);
        self.expect_on_member(member, dst);
        let overlap = src.overlap(dst);
        assert!(
            matches!(
                (overlap, in_place),
                (Overlap::Disjoint, _) | (Overlap::Same, InPlace::Allowed)
            ),
            "member {member}'s {collective} source and destination overlap ({overlap:?})"
        );
    }

    fn expect_on_member(&self, member: usize, handle: &Handle) {
        let device = self.clients[member].service_id();
        assert!(
            handle.service == device,
            "member {member}'s buffer is on {}, not on its device {device}",
            handle.service
        );
    }

    fn expect_sizes<'a>(
        collective: Collective,
        buffer: CollectiveBuffer,
        handles: impl Iterator<Item = &'a Handle>,
        expected: u64,
    ) {
        for (member, handle) in handles.enumerate() {
            let size = handle.size_in_used();
            assert!(
                size == expected,
                "member {member}'s {collective} {buffer} holds {size} bytes where it needs {expected}"
            );
        }
    }

    /// Prepares every member, then queues each member's part before anything else can queue.
    fn queue<Part>(
        &self,
        collective: Collective,
        parts: Vec<Part>,
        mut queue_part: impl FnMut(usize, &Client, Part, Vec<DeviceId>, Vote),
    ) {
        // Outside the lock, which setting the group up takes itself.
        self.clients[0].prepare_collective(collective, self.device_ids.clone());
        let votes = Agreement::new(self.clients.len()).votes();
        // Each vote waits on its device thread for its peers', so every device takes parts in one order.
        let _order = PEER_ORDER.lock();
        for (member, ((client, part), vote)) in self.clients.iter().zip(parts).zip(votes).enumerate()
        {
            queue_part(member, client, part, self.device_ids.clone(), vote);
        }
        // A part waits in its client's queue until flushed, and a member blocked reading its
        // result would wait on a peer whose part never reached the device.
        for client in &self.clients {
            client.device.flush_queue();
        }
    }
}
