//! Collectives over the clients of several devices: queued by one caller for every member at
//! once, or by each member for itself.

use super::{Client, PEER_ORDER};
use crate::logging::{CollectivePath, StreamRole};
use crate::server::{
    Collective, CollectiveBuffer, ComputeWait, DeviceTransport, Handle, Overlap, ReduceOperation,
    Reduction, Server,
};
use alloc::vec::Vec;
use cubecl_common::{device::DeviceId, device_handle::DeviceHandle};
use cubecl_ir::ElemType;
use thiserror::Error;

/// The clients of distinct devices of one runtime that run collectives together: for a caller
/// holding every member's buffers, or through [`members`](Self::members) for each member queueing
/// its own part.
///
/// Member `i` is rank `i`, in ascending device order. Work queued after a collective reads its
/// result, but per-device `all_reduce` calls are not ordered against it, and on a device both use
/// the two can hang. A member whose part fails, such as one whose source an earlier failure left
/// stale, leaves the others waiting in it and runs no later collective over the group. Buffers
/// move as bytes, so a pitched handle's row padding counts toward its chunks.
#[derive(Clone)]
pub struct ClientGroup {
    clients: Vec<Client>,
    device_ids: Vec<DeviceId>,
    transport: DeviceTransport,
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
        let mut transport = transports[0].clone();
        for member in &transports[1..] {
            transport
                .reduces
                .retain(|dtype| member.reduces.contains(dtype));
        }
        Ok(Self {
            clients,
            device_ids,
            transport,
        })
    }

    /// The members' clients, member `i` first being rank `i`.
    pub fn clients(&self) -> &[Client] {
        &self.clients
    }

    /// The transport the members share: what runs it, and the element types every member reduces.
    pub fn transport(&self) -> &DeviceTransport {
        &self.transport
    }

    /// Whether any member's device skips communication, which makes every collective over the
    /// group skip it: a member that queues its part waits on its device for every peer.
    pub fn skips_communication(&self) -> bool {
        self.clients.iter().any(Client::skips_communication)
    }

    /// Whether the members reduce elements of `dtype` together. Every other collective moves bytes,
    /// whatever their type.
    pub fn reduces(&self, dtype: ElemType) -> bool {
        self.transport.reduces.contains(&dtype)
    }

    /// Every member's `dst` receives every member's `src` reduced by `op`. A member's `src` may be
    /// its `dst`.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, buffers))
    )]
    pub fn all_reduce(&self, buffers: Vec<MemberBuffers>, dtype: ElemType, op: ReduceOperation) {
        self.exchange(Exchange::AllReduce(Reduction { dtype, op }), buffers);
    }

    /// Every member's `dst` receives every member's `src`, concatenated in member order.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, buffers))
    )]
    pub fn all_gather(&self, buffers: Vec<MemberBuffers>) {
        self.exchange(Exchange::AllGather, buffers);
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
        self.exchange(Exchange::ReduceScatter(Reduction { dtype, op }), buffers);
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
        let bytes = src.size_in_used();
        self.queue(collective, bytes, dst, |member, client, dst, device_ids| {
            let src = (member == root).then(|| src.clone());
            client.queue_broadcast(src, dst, root_id, device_ids)
        });
    }

    /// Member `i`'s `dst` receives chunk `i` of every member's `src`, concatenated in member
    /// order.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, buffers))
    )]
    pub fn all_to_all(&self, buffers: Vec<MemberBuffers>) {
        self.exchange(Exchange::AllToAll, buffers);
    }

    /// Every member, rank `i` at position `i`, each queueing its own part of the collectives over
    /// the group: for a caller that reaches each member's buffers from somewhere else, such as
    /// that device's own thread.
    ///
    /// Sets the group up on every device first, so that no part queued afterwards reaches its
    /// device before the group it runs over.
    pub fn members(&self) -> Vec<GroupMember> {
        self.clients[0].prepare_collective(Collective::CommInit, self.device_ids.clone());
        (0..self.clients.len())
            .map(|rank| GroupMember {
                group: self.clone(),
                rank,
            })
            .collect()
    }

    /// Checks every member's part of `exchange`, with sources of one size, then queues them all.
    fn exchange(&self, exchange: Exchange, buffers: Vec<MemberBuffers>) {
        let collective = exchange.collective();
        self.expect_parts(collective, buffers.len());
        let src = buffers[0].src.size_in_used();
        Self::expect_sizes(
            collective,
            CollectiveBuffer::Source,
            buffers.iter().map(|it| &it.src),
            src,
        );
        for (rank, buffers) in buffers.iter().enumerate() {
            exchange.expect_part(self, rank, buffers);
        }
        self.queue(
            collective,
            src,
            buffers,
            |_, client, buffers, device_ids| {
                exchange.queue(client, buffers, device_ids);
            },
        );
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
            Self::expect_size(collective, buffer, member, handle, expected);
        }
    }

    fn expect_size(
        collective: Collective,
        buffer: CollectiveBuffer,
        member: usize,
        handle: &Handle,
        expected: u64,
    ) {
        let size = handle.size_in_used();
        assert!(
            size == expected,
            "member {member}'s {collective} {buffer} holds {size} bytes where it needs {expected}"
        );
    }

    /// Prepares every member, then queues each member's part before anything else can queue.
    fn queue<Part>(
        &self,
        collective: Collective,
        bytes: u64,
        parts: Vec<Part>,
        mut queue_part: impl FnMut(usize, &Client, Part, Vec<DeviceId>),
    ) {
        if self.skips_communication() {
            let _skipped = self.clients[0].collective_scope(
                collective,
                &self.device_ids,
                bytes,
                CollectivePath::Skipped,
            );
            return;
        }
        // Outside the lock, which setting the group up takes itself.
        self.clients[0].prepare_collective(collective, self.device_ids.clone());
        let _collective = self.clients[0].collective_scope(
            collective,
            &self.device_ids,
            bytes,
            CollectivePath::Native(self.transport.provider),
        );
        let _order = PEER_ORDER.lock();
        for (member, (client, part)) in self.clients.iter().zip(parts).enumerate() {
            client.issue_part(StreamRole::Collective);
            queue_part(member, client, part, self.device_ids.clone());
        }
        // A part waits in its client's queue until flushed, and a member blocked reading its
        // result would wait on a peer whose part never reached the device.
        for client in &self.clients {
            client.device.flush_queue();
        }
    }
}

/// One member of a [`ClientGroup`], queueing its own part of each collective over the group.
///
/// Every member queues its part of every collective over the group, and in one order on every
/// member: the members wait for each other on their devices, so a part queued out of order, or
/// never, hangs the others. Two callers queueing over one group agree on that order themselves.
/// Each part is checked against its own member's buffers only: sources of different sizes on
/// different members fail on the devices instead.
#[derive(Clone)]
pub struct GroupMember {
    group: ClientGroup,
    rank: usize,
}

impl GroupMember {
    /// This member's rank in its group.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// This member's part of [`ClientGroup::all_reduce`].
    pub fn all_reduce(&self, buffers: MemberBuffers, dtype: ElemType, op: ReduceOperation) {
        self.exchange(Exchange::AllReduce(Reduction { dtype, op }), buffers);
    }

    /// This member's part of [`ClientGroup::all_gather`].
    pub fn all_gather(&self, buffers: MemberBuffers) {
        self.exchange(Exchange::AllGather, buffers);
    }

    /// This member's part of [`ClientGroup::reduce_scatter`].
    pub fn reduce_scatter(&self, buffers: MemberBuffers, dtype: ElemType, op: ReduceOperation) {
        self.exchange(Exchange::ReduceScatter(Reduction { dtype, op }), buffers);
    }

    /// This member's part of [`ClientGroup::all_to_all`].
    pub fn all_to_all(&self, buffers: MemberBuffers) {
        self.exchange(Exchange::AllToAll, buffers);
    }

    /// Checks and queues this member's part, then hands it to its device at once: the peers
    /// queueing theirs wait in it on their devices until it arrives.
    fn exchange(&self, exchange: Exchange, buffers: MemberBuffers) {
        exchange.expect_part(&self.group, self.rank, &buffers);
        if self.group.skips_communication() {
            return;
        }
        let client = &self.group.clients[self.rank];
        client.issue_part(StreamRole::Collective);
        exchange.queue(client, buffers, self.group.device_ids.clone());
        client.device.flush_queue();
    }
}

/// A collective every member contributes a source to and receives a destination from.
#[derive(Debug, Clone, Copy)]
enum Exchange {
    AllReduce(Reduction),
    AllGather,
    ReduceScatter(Reduction),
    AllToAll,
}

impl Exchange {
    fn collective(self) -> Collective {
        match self {
            Self::AllReduce(_) => Collective::AllReduce,
            Self::AllGather => Collective::AllGather,
            Self::ReduceScatter(_) => Collective::ReduceScatter,
            Self::AllToAll => Collective::AllToAll,
        }
    }

    /// Checks member `rank`'s buffers: on its device, apart unless the collective runs in place,
    /// and sized for its source.
    fn expect_part(self, group: &ClientGroup, rank: usize, buffers: &MemberBuffers) {
        let collective = self.collective();
        if let Self::AllReduce(reduction) | Self::ReduceScatter(reduction) = self {
            group.expect_reduces(collective, reduction.dtype);
        }
        let in_place = match self {
            Self::AllReduce(_) => InPlace::Allowed,
            _ => InPlace::Refused,
        };
        group.expect_member_buffers(collective, rank, &buffers.src, &buffers.dst, in_place);
        let src = buffers.src.size_in_used();
        let dst = match self {
            Self::AllReduce(reduction) => {
                Client::expect_whole_elements(collective, reduction.dtype, src);
                src
            }
            Self::AllGather => src * group.clients.len() as u64,
            Self::ReduceScatter(reduction) => {
                let chunk = group.expect_chunks(collective, src);
                Client::expect_whole_elements(collective, reduction.dtype, chunk);
                chunk
            }
            Self::AllToAll => {
                group.expect_chunks(collective, src);
                src
            }
        };
        ClientGroup::expect_size(
            collective,
            CollectiveBuffer::Destination,
            rank,
            &buffers.dst,
            dst,
        );
    }

    /// Queues one member's part on `client`, its compute stream waiting for it before anything
    /// queued after.
    fn queue(self, client: &Client, buffers: MemberBuffers, device_ids: Vec<DeviceId>) {
        let MemberBuffers { src, dst } = buffers;
        match self {
            Self::AllReduce(reduction) => {
                client.queue_all_reduce(src, dst, reduction, device_ids, ComputeWait::Now)
            }
            Self::AllGather => client.queue_all_gather(src, dst, device_ids),
            Self::ReduceScatter(reduction) => {
                client.queue_reduce_scatter(src, dst, reduction, device_ids)
            }
            Self::AllToAll => client.queue_all_to_all(src, dst, device_ids),
        }
    }
}
