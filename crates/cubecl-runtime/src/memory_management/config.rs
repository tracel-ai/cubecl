//! Resolving a memory configuration into the pools it asks for: value types
//! only, so the client can validate a layout before shipping it to a server.

use super::{MemoryConfiguration, MemoryPoolOptions, PoolType};
use crate::config::memory::{MemoryPoolConfig, MemoryPoolsConfig, MemoryPoolsPreset};
use alloc::vec::Vec;
use cubecl_ir::MemoryDeviceProperties;

/// Why a `memory.pools` config could not be turned into a pool layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolConfigError {
    /// `memory.pools` was an empty list.
    EmptyPoolList,
    /// A size field that must be non-zero was zero.
    ZeroSize {
        /// The offending field.
        field: &'static str,
    },
    /// `max_slice_size` exceeds `page_size` (a slice can never span pages).
    SliceLargerThanPage {
        /// The page size in bytes (after alignment).
        page_size: u64,
        /// The maximum slice size in bytes (after alignment).
        max_slice_size: u64,
    },
    /// `max_pool_size` is smaller than `page_size` (the cap can't fit one page).
    CapSmallerThanPage {
        /// The page size in bytes (after alignment).
        page_size: u64,
        /// The pool capacity in bytes.
        max_pool_size: u64,
    },
    /// `max_pool_size` spans more pages of `page_size` than a pool can hold.
    TooManyPages {
        /// The number of pages the configuration asks for.
        pages: u64,
    },
    /// The pool list has more entries than the pool routing can address.
    TooManyPools {
        /// The number of entries in the configuration.
        count: usize,
    },
    /// The preset is not available in this build.
    PresetUnavailable {
        /// The preset name.
        preset: &'static str,
    },
    /// Sliced pools are not available in this build (`exclusive_memory_only`).
    SlicedPoolsUnavailable,
}

impl core::fmt::Display for PoolConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PoolConfigError::EmptyPoolList => write!(f, "the pool list is empty"),
            PoolConfigError::ZeroSize { field } => write!(f, "`{field}` must be non-zero"),
            PoolConfigError::SliceLargerThanPage {
                page_size,
                max_slice_size,
            } => write!(
                f,
                "`max_slice_size` ({max_slice_size}) exceeds `page_size` ({page_size}); a slice can never span pages"
            ),
            PoolConfigError::CapSmallerThanPage {
                page_size,
                max_pool_size,
            } => write!(
                f,
                "`max_pool_size` ({max_pool_size}) is smaller than `page_size` ({page_size}); the cap can't fit a single page"
            ),
            PoolConfigError::TooManyPages { pages } => write!(
                f,
                "`max_pool_size` spans {pages} pages of `page_size`, exceeding the maximum of {}; increase `page_size` or lower the cap",
                u16::MAX
            ),
            PoolConfigError::TooManyPools { count } => write!(
                f,
                "the pool list has {count} entries, exceeding the maximum of {} dynamic pools",
                PERSISTENT_POOL_POS - 1
            ),
            PoolConfigError::PresetUnavailable { preset } => {
                write!(f, "the `{preset}` preset is not available in this build")
            }
            PoolConfigError::SlicedPoolsUnavailable => {
                write!(
                    f,
                    "sliced pools are not available in this build (exclusive memory only)"
                )
            }
        }
    }
}

impl core::error::Error for PoolConfigError {}

impl MemoryConfiguration {
    /// Resolve a programmatic [`MemoryPoolsConfig`] override against the
    /// runtime-chosen configuration for the **main GPU** pool.
    ///
    /// When `pools` is `None`, the runtime's own `self` is kept unchanged;
    /// when present, it wins. There is deliberately no config-file pathway for
    /// pool layouts — they are dynamic (set per model just before a load) and
    /// must not freeze at startup; the override reaches the server through
    /// [`install_memory_pools`](crate::client::Client::install_memory_pools).
    ///
    /// `page_size` is deliberately not validated against
    /// [`MemoryDeviceProperties::max_page_size`]: that value is a sizing
    /// heuristic for the default layouts (CUDA/HIP report a quarter of the
    /// device memory), not an allocation limit, and a large arena is exactly
    /// what an explicit pool override is for. An unallocatable page fails at
    /// allocation time.
    pub fn resolve(
        self,
        pools: Option<&MemoryPoolsConfig>,
        properties: &MemoryDeviceProperties,
    ) -> Result<Self, PoolConfigError> {
        let Some(pools) = pools else {
            return Ok(self);
        };

        match pools {
            MemoryPoolsConfig::Preset(MemoryPoolsPreset::SubSlices) => {
                #[cfg(exclusive_memory_only)]
                {
                    Err(PoolConfigError::PresetUnavailable {
                        preset: "sub-slices",
                    })
                }
                #[cfg(not(exclusive_memory_only))]
                {
                    Ok(MemoryConfiguration::SubSlices)
                }
            }
            MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages) => {
                Ok(MemoryConfiguration::ExclusivePages)
            }
            MemoryPoolsConfig::Explicit(entries) => {
                if entries.is_empty() {
                    return Err(PoolConfigError::EmptyPoolList);
                }
                // Slices route through their pool's position, and the
                // persistent pool owns the sentinel position, so the list must
                // stay addressable below it — checked here so the caller gets
                // the error instead of a panic on the device thread.
                if entries.len() >= PERSISTENT_POOL_POS as usize {
                    return Err(PoolConfigError::TooManyPools {
                        count: entries.len(),
                    });
                }
                let pool_options = entries
                    .iter()
                    .map(|entry| pool_options_from_entry(entry, properties))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(MemoryConfiguration::Custom { pool_options })
            }
        }
    }
}

/// Convert one config entry into runtime pool options, aligning sizes up to
/// the device alignment (a device constraint, not a user error).
fn pool_options_from_entry(
    entry: &MemoryPoolConfig,
    properties: &MemoryDeviceProperties,
) -> Result<MemoryPoolOptions, PoolConfigError> {
    let alignment = properties.alignment.max(1);
    match entry {
        MemoryPoolConfig::Exclusive {
            max_alloc_size,
            dealloc_period,
        } => {
            // 0 stays 0: a pool dedicated to zero-sized allocations, as used by
            // the `SubSlices` preset.
            let max_alloc_size = max_alloc_size.bytes().next_multiple_of(alignment);
            Ok(MemoryPoolOptions {
                pool_type: PoolType::ExclusivePages { max_alloc_size },
                dealloc_period: *dealloc_period,
            })
        }
        MemoryPoolConfig::Direct { reclaim_at } => Ok(MemoryPoolOptions {
            pool_type: PoolType::Direct {
                reclaim_at: reclaim_at.map(|size| size.bytes()),
            },
            // `dealloc_period` has no meaning here: the pool reclaims on
            // memory pressure, not on an allocation count.
            dealloc_period: None,
        }),
        // Sliced pools break the invariant `exclusive_memory_only` builds rely
        // on (e.g. wgpu on wasm assumes a buffer is never shared between
        // slices), so an explicit list must be rejected just like the
        // `sub-slices` preset is.
        #[cfg(exclusive_memory_only)]
        MemoryPoolConfig::Sliced { .. } => Err(PoolConfigError::SlicedPoolsUnavailable),
        #[cfg(not(exclusive_memory_only))]
        MemoryPoolConfig::Sliced {
            page_size,
            max_slice_size,
            max_pool_size,
            dealloc_period,
        } => {
            if page_size.bytes() == 0 {
                return Err(PoolConfigError::ZeroSize { field: "page_size" });
            }

            let page_size = page_size.bytes().next_multiple_of(alignment);
            let max_slice_size = match max_slice_size {
                Some(size) if size.bytes() == 0 => {
                    return Err(PoolConfigError::ZeroSize {
                        field: "max_slice_size",
                    });
                }
                Some(size) => size.bytes().next_multiple_of(alignment),
                None => page_size,
            };
            if max_slice_size > page_size {
                return Err(PoolConfigError::SliceLargerThanPage {
                    page_size,
                    max_slice_size,
                });
            }
            if let Some(cap) = max_pool_size {
                let cap = cap.bytes();
                if cap == 0 {
                    return Err(PoolConfigError::ZeroSize {
                        field: "max_pool_size",
                    });
                }
                if cap < page_size {
                    return Err(PoolConfigError::CapSmallerThanPage {
                        page_size,
                        max_pool_size: cap,
                    });
                }
                let pages = cap / page_size;
                if pages > u16::MAX as u64 {
                    return Err(PoolConfigError::TooManyPages { pages });
                }
            }

            Ok(MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size,
                    max_slice_size,
                    max_pool_size: max_pool_size.map(|size| size.bytes()),
                },
                dealloc_period: *dealloc_period,
            })
        }
    }
}

/// The pool position stamped on persistent-pool slices, routing their binds
/// and lookups to the persistent pool. A fixed sentinel (rather than "one past
/// the dynamic pools") so live persistent slices stay routable when
/// [`MemoryManagement::install_pools`] rebuilds the dynamic pools with a
/// different count.
#[doc(hidden)]
pub const PERSISTENT_POOL_POS: u8 = u8::MAX;
