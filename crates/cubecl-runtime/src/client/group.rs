//! Collectives queued by one caller over the clients of several devices.

use super::{Client, PEER_ORDER};
use crate::server::{Handle, ReduceOperation};
use alloc::vec::Vec;
use cubecl_common::device::DeviceId;
use cubecl_ir::ElemType;
use thiserror::Error;

/// The clients of distinct devices of one runtime that run collectives together, for a caller
/// holding every member's buffers.
///
/// Member `i` is rank `i`, so the members are in ascending device order, the order a collective's
/// ranks take. A collective queues every member's part before any other caller queues one, so
/// every device sees collectives in one order, and returns once each member's stream waits for
/// its part.
#[derive(Clone)]
pub struct ClientGroup {
    clients: Vec<Client>,
    device_ids: Vec<DeviceId>,
}

/// One member's buffers in a collective.
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
        self.queue(buffers, |client, buffers, device_ids| {
            client.all_reduce(buffers.src, buffers.dst, dtype, device_ids, op)
        });
    }

    /// Every member's `dst` receives every member's `src`, concatenated in member order.
    pub fn all_gather(&mut self, buffers: Vec<MemberBuffers>, dtype: ElemType) {
        self.queue(buffers, |client, buffers, device_ids| {
            client.all_gather(buffers.src, buffers.dst, dtype, device_ids)
        });
    }

    /// Member `i`'s `dst` receives chunk `i` of every member's `src` reduced by `op`.
    pub fn reduce_scatter(
        &mut self,
        buffers: Vec<MemberBuffers>,
        dtype: ElemType,
        op: ReduceOperation,
    ) {
        self.queue(buffers, |client, buffers, device_ids| {
            client.reduce_scatter(buffers.src, buffers.dst, dtype, device_ids, op)
        });
    }

    /// Every member's `dst` receives `src`, which member `root` holds.
    pub fn broadcast(&mut self, root: usize, src: Handle, dst: Vec<Handle>, dtype: ElemType) {
        let root_id = self.device_ids[root];
        let mut src = Some(src);
        let parts = dst.into_iter().enumerate().collect();
        self.queue(parts, |client, (member, dst), device_ids| {
            let src = match member == root {
                true => src.take(),
                false => None,
            };
            client.broadcast(src, dst, dtype, root_id, device_ids)
        });
    }

    /// Member `i`'s `dst` receives chunk `i` of every member's `src`, concatenated in member
    /// order.
    pub fn all_to_all(&mut self, buffers: Vec<MemberBuffers>, dtype: ElemType) {
        self.queue(buffers, |client, buffers, device_ids| {
            client.all_to_all(buffers.src, buffers.dst, dtype, device_ids)
        });
    }

    /// Queues each member's part, then makes each member's stream wait for it.
    fn queue<Part>(
        &mut self,
        parts: Vec<Part>,
        mut queue_part: impl FnMut(&mut Client, Part, Vec<DeviceId>),
    ) {
        assert_eq!(
            parts.len(),
            self.clients.len(),
            "a collective takes one part per member"
        );
        // Outside the lock, which a first setup takes itself.
        self.clients[0].ensure_init_collective(self.device_ids.clone());
        {
            let _order = PEER_ORDER.lock();
            for (client, part) in self.clients.iter_mut().zip(parts) {
                queue_part(client, part, self.device_ids.clone());
            }
        }
        self.clients.iter().for_each(Client::sync_collective);
    }
}
