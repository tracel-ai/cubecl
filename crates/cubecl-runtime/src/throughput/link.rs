use crate::{logging::TransferRoute, server::Collective};

/// How a link probe moves its bytes between devices.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum LinkRoute {
    /// `Client::to_client` from the first member to the second, over the route the runtime takes
    /// between them: a rate over the device transport says nothing about a copy through the host.
    Transfer(TransferRoute),
    /// A kernel on the second member loading from the first member's memory.
    PeerLoad,
    /// A kernel on the first member storing into the second member's memory, ending behind a
    /// system-scope fence: a store is posted, and without one the kernel ends before its bytes
    /// arrive.
    PeerStore,
    /// A collective over every member. Sized as nccl-tests sizes it: the per-member buffer for an
    /// all-reduce, the whole gathered output for an all-gather, the whole reduced input for a
    /// reduce-scatter.
    Collective(Collective),
    /// Every member storing into the next member's memory at once, the last into the first's, each
    /// kernel ending behind a system-scope fence as [`PeerStore`](Self::PeerStore) does. Sized per
    /// member: the bytes each one stores.
    RingPeerStore,
    /// Every member loading from the previous member's memory at once, the first from the last's.
    /// Sized per member: the bytes each one loads.
    RingPeerLoad,
}

/// What a link probe measures, between the members its cache is named for.
///
/// A link's rate belongs to the cards and the path between them, so the members are not part of
/// the key: they name the cache, as a device names its [`ThroughputCache`](super::ThroughputCache).
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(serializable, serde(deny_unknown_fields))]
pub struct LinkKey {
    /// How the bytes move.
    pub route: LinkRoute,
    /// The bytes one pass moves, as the route counts them.
    pub bytes: u64,
}
