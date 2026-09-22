//! Where the pools that are not the dynamic ones live.

/// The pool position stamped on persistent-pool slices, routing their binds
/// and lookups to the persistent pool. A fixed sentinel (rather than "one past
/// the dynamic pools") so a persistent slice stays routable however many
/// dynamic pools a workload leaves behind.
#[doc(hidden)]
pub const PERSISTENT_POOL_POS: u8 = u8::MAX;

/// The pool position stamped on dedicated allocations
/// ([`MemoryAllocationMode::Dedicated`](super::MemoryAllocationMode::Dedicated)),
/// a fixed sentinel for the same reason as [`PERSISTENT_POOL_POS`].
#[doc(hidden)]
pub const DEDICATED_POOL_POS: u8 = u8::MAX - 1;
