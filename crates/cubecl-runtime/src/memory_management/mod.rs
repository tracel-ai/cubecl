mod base;
mod config;
mod handle;

pub use base::*;
pub use config::*;
pub use handle::*;

use alloc::vec::Vec;

/// The type of memory pool to use.
#[derive(Debug, Clone)]
pub enum PoolType {
    /// Use a memory where every allocation is a separate page.
    ExclusivePages {
        /// The minimum number of bytes to allocate in this pool.
        max_alloc_size: u64,
    },
    /// Give every allocation its own device allocation, sized to the request
    /// and reused by exact size.
    ///
    /// No carving at all, so it wastes only alignment padding. Worth it where
    /// padding matters more than allocation count — under a
    /// [`DryRun`](crate::dry_run::DryRun), where unresolved reservations never
    /// reach the driver at all, or on a device the workload barely fits.
    Direct {
        /// Reserved bytes above which free slices are returned to the driver.
        /// A watermark rather than a budget; `None` reclaims only on an
        /// explicit cleanup.
        reclaim_at: Option<u64>,
    },
    /// Use a memory where each allocation is a slice of a bigger allocation.
    SlicedPages {
        /// The page size to allocate.
        page_size: u64,
        /// The maximum size of a slice to allocate in the pool.
        max_slice_size: u64,
        /// Hard cap on the total bytes of pages this pool may hold.
        ///
        /// The effective cap is `floor(max_pool_size / page_size)` whole pages.
        /// If `max_pool_size < page_size`, the page size is shrunk to the
        /// (alignment-rounded) cap so the budget is honored with a single page.
        /// When the cap is reached and no free slice fits after coalescing,
        /// reserving returns [`IoError`](crate::server::IoError)
        /// `PoolCapacityExceeded` instead of silently growing. `None` (the
        /// previous behavior) keeps unbounded growth.
        ///
        /// Note: runtimes that create one memory management per stream (CUDA,
        /// HIP) apply the cap per stream.
        max_pool_size: Option<u64>,
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

#[derive(Default, Clone, Copy, Debug)]
/// The mode of allocation used.
pub enum MemoryAllocationMode {
    /// Use the automatic memory management strategy for allocation.
    #[default]
    Auto,
    /// Use a persistent memory management strategy, meaning that all allocations are for data that is
    /// likely never going to be freed.
    Persistent,
}

/// Why installing a dynamic pool layout did not take effect.
///
/// The layout itself was already valid — that is
/// [`PoolConfigError`](PoolConfigError), reported when the configuration is
/// resolved. This is about the pools' *state* at the moment of the swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMemoryPoolsError {
    /// The dynamic pools still hold live allocations, so the old layout was
    /// kept. A live slice carries its pool position, and swapping the pool
    /// list under it would leave that position pointing at a different pool.
    ///
    /// Transient: retry once whatever holds them drains. A cleanup that does
    /// not clear it usually means a cache is holding slices (the metadata
    /// info cache) or a captured graph is pinning them.
    PoolsInUse {
        /// Bytes still live in the dynamic pools.
        bytes_in_use: u64,
    },
    /// This server has no configurable dynamic pools. Permanent — unlike
    /// [`PoolsInUse`](Self::PoolsInUse), retrying will never succeed.
    Unsupported,
}

impl core::fmt::Display for InstallMemoryPoolsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InstallMemoryPoolsError::PoolsInUse { bytes_in_use } => write!(
                f,
                "the dynamic pools kept their layout: {bytes_in_use} bytes are still live in them"
            ),
            InstallMemoryPoolsError::Unsupported => {
                write!(f, "this server has no configurable dynamic memory pools")
            }
        }
    }
}

impl core::error::Error for InstallMemoryPoolsError {}
