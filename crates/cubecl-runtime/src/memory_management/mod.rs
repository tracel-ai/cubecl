pub(crate) mod memory_pool;

mod base;
mod memory_lock;

pub use base::*;
pub use memory_lock::*;

/// Dynamic memory management strategy.
mod memory_manage;
pub use memory_manage::*;

/// The type of memory pool to use.
#[derive(Debug, Clone)]
pub enum PoolType {
    /// Use a memory where every allocation is a separate page.
    ExclusivePages,
    /// Use a memory where each allocation is a slice of a bigger allocation.
    SlicedPages {
        /// The maxiumum size of a slice to allocate in the pool.
        max_slice_size: usize,
    },
}

/// Options to create a memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolOptions {
    /// What kind of pool to use.
    pub pool_type: PoolType,
    /// The amount of bytes used for each chunk in the memory pool.
    pub page_size: usize,
    /// The number of chunks allocated directly at creation.
    ///
    /// Useful when you know in advance how much memory you'll need.
    pub chunk_num_prealloc: usize,
}

/// High level configuration of memory management.
#[derive(Clone, Debug)]
pub enum MemoryConfiguration {
    /// The default preset using sub sices.
    #[cfg(not(exclusive_memory_only))]
    SubSlices,
    /// Default preset using only exclusive pages.
    /// This can be necessary when backends don't support sub-slices.
    ExclusivePages,
    /// Customize each pool individually.
    Custom(Vec<MemoryPoolOptions>),
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

/// Properties of the device related to allocation.
#[derive(Debug, Clone)]
pub struct MemoryDeviceProperties {
    /// The maximum nr. of bytes that can be allocated in one go.
    pub max_page_size: usize,
    /// The required memory offset alignment in bytes.
    pub alignment: usize,
}
