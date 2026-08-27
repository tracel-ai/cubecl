//! Collectives across devices: the groups this device has joined, and what a
//! driver has to supply to run one.
//!
//! The bookkeeping is the same whichever library is underneath. A group is
//! named by the devices in it, every rank joins under one identifier the group
//! agrees on, and a device's rank is its position in the sorted list — get
//! that wrong on one rank and the whole group hangs. That is what lives here.
//!
//! What does not is [`CollectiveDriver`]: agreeing the identifier, joining,
//! naming an element type, and the three operations themselves.

use super::{DeviceResource, Driver};
use crate::server::{CommunicationId, ReduceOperation, ServerError};
use alloc::format;
use alloc::vec::Vec;
use cubecl_common::device::DeviceId;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_ir::ElemType;

/// A driver that can run collectives across devices.
pub trait CollectiveDriver: Driver {
    /// This device's membership of one group, joined once and kept.
    type Communicator;
    /// The identifier every rank of a group joins under.
    type UniqueId: Copy;
    /// How the driver names an element type.
    type DataType: Copy;
    /// The stream collectives are issued on, kept apart from the compute
    /// streams so a collective never blocks one.
    type CommStream: Copy;

    /// The identifier the group `id` names joins under.
    ///
    /// Minted by whichever rank asks first and remembered for the rest, so
    /// this is process-wide state the driver keeps: the servers of two devices
    /// in one group are two objects that have to agree on one answer.
    ///
    /// # Errors
    ///
    /// The driver's refusal to mint one, which stops the group forming at all.
    fn group_id(id: &CommunicationId) -> Result<Self::UniqueId, ServerError>;

    /// Join the group `id` names as rank `rank` of `ranks`.
    ///
    /// # Errors
    ///
    /// The driver's refusal to join, which every other rank sees as this one
    /// never arriving.
    fn join(
        id: Self::UniqueId,
        ranks: usize,
        rank: usize,
    ) -> Result<Self::Communicator, ServerError>;

    /// How the driver names `dtype`, and how many elements `size` bytes hold.
    ///
    /// # Errors
    ///
    /// An element type the driver has no name for. Reported rather than fatal:
    /// a collective is one operation among many, and refusing it is not a
    /// reason to take the process down — the caller can pick another type, or
    /// another way to move the tensor.
    fn data_type(dtype: ElemType, size: u64) -> Result<(Self::DataType, usize), ServerError>;

    /// Reduce `src` across the group into `dst` on every rank.
    ///
    /// # Errors
    ///
    /// The driver's refusal to enqueue the reduction.
    fn all_reduce(
        comm: &Self::Communicator,
        src: &DeviceResource<Self>,
        dst: &DeviceResource<Self>,
        dtype: Self::DataType,
        count: usize,
        op: ReduceOperation,
        stream: Self::CommStream,
    ) -> Result<(), ServerError>;

    /// Send `src` to `peer`.
    ///
    /// # Errors
    ///
    /// The driver's refusal to enqueue the send.
    fn send(
        comm: &Self::Communicator,
        src: &DeviceResource<Self>,
        dtype: Self::DataType,
        count: usize,
        peer: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError>;

    /// Receive into `dst` from `peer`.
    ///
    /// # Errors
    ///
    /// The driver's refusal to enqueue the receive.
    fn recv(
        comm: &Self::Communicator,
        dst: &DeviceResource<Self>,
        dtype: Self::DataType,
        count: usize,
        peer: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError>;
}

/// The groups this device has joined.
///
/// One per server, unlike the identifiers behind
/// [`group_id`](CollectiveDriver::group_id): an identifier is what every rank
/// of a group agrees on, a communicator is one rank's membership of it.
pub struct Collectives<D: CollectiveDriver> {
    /// This device, whose position in a sorted group is its rank.
    device: DeviceId,
    joined: HashMap<CommunicationId, D::Communicator>,
}

impl<D: CollectiveDriver> core::fmt::Debug for Collectives<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The communicators are opaque driver handles, so which groups this
        // device is in is the whole of what there is to say.
        f.debug_struct("Collectives")
            .field("device", &self.device)
            .field("joined", &self.joined.len())
            .finish()
    }
}

impl<D: CollectiveDriver> Collectives<D> {
    /// A device that has joined nothing yet.
    pub fn new(device: DeviceId) -> Self {
        Self {
            device,
            joined: HashMap::default(),
        }
    }

    /// Join the group over `devices`, and answer with the id it is known by.
    ///
    /// A group already joined is joined once: `Ok(None)` says there was
    /// nothing to do, so the caller does not announce a membership twice.
    ///
    /// # Errors
    ///
    /// [`ServerError::Generic`] when this device is not among `devices` — it
    /// would have no rank in a group it is not in — and whatever the driver
    /// says about agreeing an identifier or joining.
    pub fn join(&mut self, devices: Vec<DeviceId>) -> Result<Option<CommunicationId>, ServerError> {
        let id = CommunicationId::from(devices.clone());
        if self.joined.contains_key(&id) {
            return Ok(None);
        }
        // Sorted, because a device's rank is its position and every rank has
        // to compute the same one. Two ranks disagreeing does not fail: it
        // hangs, with each waiting for a peer that is answering to a different
        // number.
        let mut devices = devices;
        devices.sort();
        let rank = self.rank_in(&devices).ok_or_else(|| ServerError::Generic {
            reason: format!(
                "this device ({:?}) is not among the {} the group was formed over, \
                     so it has no rank in it",
                self.device,
                devices.len()
            ),
            backtrace: BackTrace::capture(),
        })?;

        let comm = D::join(D::group_id(&id)?, devices.len(), rank)?;
        self.joined.insert(id.clone(), comm);
        Ok(Some(id))
    }

    /// This device's membership of the group over `devices`.
    ///
    /// # Errors
    ///
    /// [`ServerError::Generic`] when this device never joined that group, which
    /// is a missing [`join`](Self::join) rather than anything the device did.
    pub fn get(&self, devices: &CommunicationId) -> Result<&D::Communicator, ServerError> {
        self.joined
            .get(devices)
            .ok_or_else(|| ServerError::Generic {
                reason: "no communicator for this group; it has to be joined first".into(),
                backtrace: BackTrace::capture(),
            })
    }

    /// The rank of the one device in `devices` that is not this one.
    ///
    /// For the two-device operations — a send has exactly one peer, and its
    /// rank is whichever position this device does not occupy.
    ///
    /// # Errors
    ///
    /// [`ServerError::Generic`] when every device in the pair is this one, so
    /// there is no peer to name.
    pub fn peer_rank(&self, devices: &[DeviceId]) -> Result<usize, ServerError> {
        devices
            .iter()
            .position(|id| id.index_id != self.device.index_id)
            .ok_or_else(|| ServerError::Generic {
                reason: format!(
                    "every device in the pair is this one ({:?}), so there is no peer",
                    self.device
                ),
                backtrace: BackTrace::capture(),
            })
    }

    /// This device's position in `devices`, which is its rank.
    fn rank_in(&self, devices: &[DeviceId]) -> Option<usize> {
        devices
            .iter()
            .position(|id| id.index_id == self.device.index_id)
    }
}
