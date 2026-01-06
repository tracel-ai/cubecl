pub(crate) mod memory_pool;

mod base;

pub use base::*;

/// Dynamic memory management strategy.
mod memory_manage;
pub use memory_manage::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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
    /// The default preset, which uses pools that allocate sub slices.
    #[cfg(not(exclusive_memory_only))]
    SubSlices,
    /// Default preset for using exclusive pages.
    /// This can be necessary for backends don't support sub-slices.
    ExclusivePages,
    /// Custom settings.
    Custom {
        /// Options for each pool to construct. When allocating, the first
        /// possible pool will be picked for an allocation.
        pool_options: Vec<MemoryPoolOptions>,
    },
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
            MemoryConfiguration::SubSlices
        }
    }
}
