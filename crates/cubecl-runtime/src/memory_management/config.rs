//! How memory is managed, and where the pools that are not the dynamic ones
//! live.

/// Whether this build refuses pools that share a page between allocations —
/// the `exclusive-memory-only` feature, or a wasm target.
///
/// Decided by this crate alone, since [`MemoryConfiguration::Adaptive`] only
/// exists under it; a crate that needs to know reads it here instead of
/// deciding again.
pub const EXCLUSIVE_MEMORY_ONLY: bool = cfg!(exclusive_memory_only);

/// High level configuration of memory management.
#[derive(Clone, Debug)]
pub enum MemoryConfiguration {
    /// One page per allocation, in exponentially spaced size buckets: what a
    /// device that cannot sub-slice gets, and what a staging or uniform pool
    /// wants whatever the device.
    ExclusivePages,
    /// Small allocations in a sliced pool of their own, everything else carved
    /// from pages sized to the largest allocation served so far — nothing to
    /// measure or configure per workload. When an allocation outgrows the
    /// pages, pages of the new size take over and the old ones are returned
    /// as they empty, or sooner when what lives on them is relocated. The
    /// default where sub-slicing is available.
    #[cfg(not(exclusive_memory_only))]
    Adaptive,
}

#[allow(clippy::derivable_impls)]
impl Default for MemoryConfiguration {
    fn default() -> Self {
        #[cfg(exclusive_memory_only)]
        {
            MemoryConfiguration::ExclusivePages
        }
        #[cfg(not(exclusive_memory_only))]
        {
            MemoryConfiguration::Adaptive
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
/// The mode of allocation used.
pub enum MemoryAllocationMode {
    /// Use the automatic memory management strategy for allocation.
    #[default]
    Auto,
    /// Use a persistent memory management strategy, meaning that all allocations are for data that is
    /// likely never going to be freed.
    Persistent,
    /// Give every allocation its own device allocation, returned to the driver
    /// once it is freed, outside every pool: for a buffer that exists only for
    /// one measurement (the memory-bandwidth probe) and must neither stay
    /// reserved nor shape the pools' sizing.
    Dedicated,
}

/// The pool position stamped on persistent-pool slices, routing their binds
/// and lookups to the persistent pool. A fixed sentinel (rather than "one past
/// the dynamic pools") so a persistent slice stays routable however many
/// dynamic pools a workload leaves behind.
#[doc(hidden)]
pub const PERSISTENT_POOL_POS: u8 = u8::MAX;

/// The pool position stamped on dedicated allocations
/// ([`MemoryAllocationMode::Dedicated`]),
/// a fixed sentinel for the same reason as [`PERSISTENT_POOL_POS`].
#[doc(hidden)]
pub const DEDICATED_POOL_POS: u8 = u8::MAX - 1;
