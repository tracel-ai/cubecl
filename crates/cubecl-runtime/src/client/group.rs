//! Collectives queued by one caller over the clients of several devices.

use super::{Client, PEER_ORDER};
use crate::server::{Collective, Handle, ReduceOperation};
use alloc::vec::Vec;
use cubecl_common::device::DeviceId;
use cubecl_ir::ElemType;
use thiserror::Error;

/// The clients of distinct devices of one runtime that run collectives together, for a caller
/// holding every member's buffers.
///
/// Member `i` is rank `i`, so the members are in ascending device order, the order a collective's
/// ranks take. A collective queues every member's part before another group or a transfer queues
/// anything, and returns once each member's stream waits for its part. Per-device collectives
/// other threads queue on the same devices are not ordered against it.
///
/// Every member's buffers are checked before any part is queued: a part one device refuses
/// leaves the devices that queued theirs waiting on it forever.
#[derive(Clone)]
pub struct ClientGroup {
    clients: Vec<Client>,
    device_ids: Vec<DeviceId>,
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

impl ClientGroup {
    /// A group of `clients`, one per member.
    ///
    /// # Errors
    ///
    /// When there are none, when they mix runtimes, when their devices are not in strictly
    /// ascending order, and when their runtime has no device transport.
    pub fn new(clients: Vec<Client>) -> Result<Self, ClientGroupError> {
        let first = clients.first().ok_or(ClientGroupError::Empty)?;
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
        if !first.has_device_transport() {
            return Err(ClientGroupError::NoDeviceTransport {
                runtime: first.name(),
            });
        }
        Ok(Self {
            clients,
            device_ids,
        })
    }

    /// The members' clients, member `i` first being rank `i`.
    pub fn clients(&self) -> &[Client] {
        &self.clients
    }

    /// Every member's `dst` receives every member's `src` reduced by `op`.
    pub fn all_reduce(
        &mut self,
        buffers: Vec<MemberBuffers>,
        dtype: ElemType,
        op: ReduceOperation,
    ) {
        let collective = Collective::AllReduce;
        let src = self.expect_buffers(collective, dtype, &buffers);
        Self::expect_sizes(
            collective,
            "destination",
            buffers.iter().map(|it| &it.dst),
            src,
        );
        self.queue(collective, buffers, |client, buffers, device_ids| {
            client.queue_all_reduce(buffers.src, buffers.dst, dtype, device_ids, op)
        });
        // An all_reduce leaves the wait to `sync_collective`, so many can overlap compute.
        self.clients.iter().for_each(Client::sync_collective);
    }

    /// Every member's `dst` receives every member's `src`, concatenated in member order.
    pub fn all_gather(&mut self, buffers: Vec<MemberBuffers>, dtype: ElemType) {
        let collective = Collective::AllGather;
        let src = self.expect_buffers(collective, dtype, &buffers);
        let members = self.clients.len() as u64;
        Self::expect_sizes(
            collective,
            "destination",
            buffers.iter().map(|it| &it.dst),
            src * members,
        );
        self.queue(collective, buffers, |client, buffers, device_ids| {
            client.queue_all_gather(buffers.src, buffers.dst, dtype, device_ids)
        });
    }

    /// Member `i`'s `dst` receives chunk `i` of every member's `src` reduced by `op`.
    pub fn reduce_scatter(
        &mut self,
        buffers: Vec<MemberBuffers>,
        dtype: ElemType,
        op: ReduceOperation,
    ) {
        let collective = Collective::ReduceScatter;
        let src = self.expect_buffers(collective, dtype, &buffers);
        let chunk = self.expect_chunks(collective, dtype, src);
        Self::expect_sizes(
            collective,
            "destination",
            buffers.iter().map(|it| &it.dst),
            chunk,
        );
        self.queue(collective, buffers, |client, buffers, device_ids| {
            client.queue_reduce_scatter(buffers.src, buffers.dst, dtype, device_ids, op)
        });
    }

    /// Every member's `dst` receives `src`, which member `root` holds.
    pub fn broadcast(&mut self, root: usize, src: Handle, dst: Vec<Handle>, dtype: ElemType) {
        let collective = Collective::Broadcast;
        let members = self.clients.len();
        assert!(
            root < members,
            "the root of a broadcast is member {root} of a group of {members}"
        );
        self.expect_parts(collective, dst.len());
        self.expect_transport_takes(collective, dtype);
        self.expect_on_member(root, &src);
        for (member, dst) in dst.iter().enumerate() {
            self.expect_on_member(member, dst);
        }
        Self::expect_sizes(collective, "destination", dst.iter(), src.size_in_used());

        let root_id = self.device_ids[root];
        let mut src = Some(src);
        let parts = dst.into_iter().enumerate().collect();
        self.queue(collective, parts, |client, (member, dst), device_ids| {
            let src = match member == root {
                true => src.take(),
                false => None,
            };
            client.queue_broadcast(src, dst, dtype, root_id, device_ids)
        });
    }

    /// Member `i`'s `dst` receives chunk `i` of every member's `src`, concatenated in member
    /// order. A member's `src` and `dst` are distinct buffers.
    pub fn all_to_all(&mut self, buffers: Vec<MemberBuffers>, dtype: ElemType) {
        let collective = Collective::AllToAll;
        let src = self.expect_buffers(collective, dtype, &buffers);
        self.expect_chunks(collective, dtype, src);
        Self::expect_sizes(
            collective,
            "destination",
            buffers.iter().map(|it| &it.dst),
            src,
        );
        for (member, buffers) in buffers.iter().enumerate() {
            assert!(
                buffers.src.memory.descriptor().id != buffers.dst.memory.descriptor().id,
                "member {member}'s all_to_all cannot run in place"
            );
        }
        self.queue(collective, buffers, |client, buffers, device_ids| {
            client.queue_all_to_all(buffers.src, buffers.dst, dtype, device_ids)
        });
    }

    /// Checks one part per member, each member's buffers on its device, sources of one size,
    /// and a type the transport moves; returns the size of each source.
    fn expect_buffers(
        &self,
        collective: Collective,
        dtype: ElemType,
        buffers: &[MemberBuffers],
    ) -> u64 {
        self.expect_parts(collective, buffers.len());
        self.expect_transport_takes(collective, dtype);
        for (member, buffers) in buffers.iter().enumerate() {
            self.expect_on_member(member, &buffers.src);
            self.expect_on_member(member, &buffers.dst);
        }
        let src = buffers[0].src.size_in_used();
        Self::expect_sizes(collective, "source", buffers.iter().map(|it| &it.src), src);
        src
    }

    fn expect_parts(&self, collective: Collective, parts: usize) {
        let members = self.clients.len();
        assert!(
            parts == members,
            "a {collective} over {members} members takes one part each, not {parts}"
        );
    }

    fn expect_transport_takes(&self, collective: Collective, dtype: ElemType) {
        assert!(
            self.clients[0].device_transport_takes(dtype),
            "a {collective} on {} cannot move {dtype:?}",
            self.clients[0].name()
        );
    }

    /// Checks `size` bytes split into one chunk of whole elements per member, and returns the
    /// size of a chunk.
    fn expect_chunks(&self, collective: Collective, dtype: ElemType, size: u64) -> u64 {
        let members = self.clients.len() as u64;
        let element = dtype.size() as u64;
        assert!(
            size.is_multiple_of(members * element),
            "a {collective} over {members} members cannot split {size} bytes of {dtype:?}"
        );
        size / members
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
        buffer: &str,
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
        &mut self,
        collective: Collective,
        parts: Vec<Part>,
        mut queue_part: impl FnMut(&Client, Part, Vec<DeviceId>),
    ) {
        // Outside the lock, which setting the group up takes itself.
        self.clients[0].prepare_collective(collective, self.device_ids.clone());
        let _order = PEER_ORDER.lock();
        for (client, part) in self.clients.iter().zip(parts) {
            queue_part(client, part, self.device_ids.clone());
        }
        // A part waits in its client's queue until flushed, and a member blocked reading its
        // result would wait on a peer whose part never reached the device.
        for client in &self.clients {
            client.device.flush_queue();
        }
    }
}
