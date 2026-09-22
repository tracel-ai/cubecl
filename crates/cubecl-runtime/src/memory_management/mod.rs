mod base;
mod config;
mod handle;
mod layout;

pub use base::*;
pub use config::*;
pub use handle::*;
pub use layout::*;

/// The type of memory pool to use.
#[derive(Debug, Clone)]
pub enum PoolType {
    /// Use a memory where every allocation is a separate page.
    ExclusivePages {
        /// The minimum number of bytes to allocate in this pool.
        max_alloc_size: u64,
    },
    /// Use a memory where each allocation is a slice of a bigger allocation.
    SlicedPages {
        /// The page size to allocate.
        page_size: u64,
        /// The maximum size of a slice to allocate in the pool.
        max_slice_size: u64,
    },
    /// Slices carved from pages whose size follows the largest allocation the
    /// pool has served: `largest + 1 MiB`, MiB-rounded, never below
    /// `min_page_size`.
    ///
    /// A page is sized once, when it is allocated. When a larger allocation
    /// raises the target, every page of the old size becomes *outdated*: it
    /// serves no new reservation and is returned to the driver as soon as its
    /// last slice is freed. An explicit cleanup — which is also the retry after
    /// a failed device allocation — moves what is still live on outdated pages
    /// onto pages of the current size, so they can be returned at once instead
    /// of waiting on their longest-lived slice.
    ///
    /// Accepts every size: a page is always large enough for what it serves.
    /// A [`SlicedPages`](PoolType::SlicedPages) pool listed before it then
    /// accepts only up to its `max_slice_size`, not also allocations close to
    /// its page size: those land here, on pages sized to them.
    AdaptivePages {
        /// The smallest page the pool allocates, so a pool whose largest
        /// allocation is small still carves pages worth carving.
        min_page_size: u64,
    },
}

/// Options to create a memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolOptions {
    /// What kind of pool to use.
    pub pool_type: PoolType,
    /// Period after which allocations are deemed unused and deallocated.
    ///
    /// This period is measured in the number of allocations in the parent allocator. If a page
    /// in the pool was unused for the entire period, it will be deallocated. This period is
    /// approximmate, as checks are only done occasionally.
    pub dealloc_period: Option<u64>,
}

/// High level configuration of memory management.
#[derive(Clone, Debug)]
pub enum MemoryConfiguration {
    /// One page per allocation, in exponentially spaced size buckets: what a
    /// device that cannot sub-slice gets, and what a staging or uniform pool
    /// wants whatever the device.
    ExclusivePages,
    /// Small allocations in a sliced pool of their own, everything else in one
    /// [`AdaptivePages`](PoolType::AdaptivePages) pool that sizes its pages
    /// from the allocations it serves — nothing to measure or configure per
    /// workload. The default where sub-slicing is available.
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
