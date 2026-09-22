//! How each [`MemoryConfiguration`] lays its pools out on a device.
//!
//! Here rather than in the server that builds the pools, because the presets
//! are only what they are under this crate's `exclusive_memory_only` cfg: a
//! crate deciding that for itself disagrees with this one as soon as the
//! feature is turned on here directly, and names a preset that is not there.

use super::{MemoryConfiguration, MemoryPoolOptions, PoolType};
#[cfg(not(exclusive_memory_only))]
use alloc::vec;
use alloc::vec::Vec;
use cubecl_ir::MemoryDeviceProperties;

/// Whether this build refuses pools that share a page between allocations —
/// the `exclusive-memory-only` feature, or a wasm target.
///
/// Decided by this crate alone, for the reason the module gives; a crate that
/// needs to know reads it here instead of deciding again.
pub const EXCLUSIVE_MEMORY_ONLY: bool = cfg!(exclusive_memory_only);

impl MemoryConfiguration {
    /// The pools this configuration lays out on a device with `properties`,
    /// in the order an allocation tries them.
    pub fn pool_options(self, properties: &MemoryDeviceProperties) -> Vec<MemoryPoolOptions> {
        match self {
            #[cfg(not(exclusive_memory_only))]
            MemoryConfiguration::Adaptive => {
                let alignment = properties.alignment;

                vec![
                    // Allocations smaller than the alignment can't use offsets
                    // at all (on wgpu at least).
                    MemoryPoolOptions {
                        pool_type: PoolType::ExclusivePages { max_alloc_size: 0 },
                        dealloc_period: None,
                    },
                    // Kernel metadata — shapes, strides, scalars — churns
                    // thousands of tiny slices. Kept off the adaptive pages so
                    // they neither fragment them nor count toward their size.
                    MemoryPoolOptions {
                        pool_type: PoolType::SlicedPages {
                            page_size: ADAPTIVE_SMALL_PAGE.next_multiple_of(alignment),
                            max_slice_size: ADAPTIVE_SMALL_SLICE.next_multiple_of(alignment),
                        },
                        dealloc_period: None,
                    },
                    MemoryPoolOptions {
                        pool_type: PoolType::AdaptivePages {
                            min_page_size: ADAPTIVE_MIN_PAGE
                                .min(properties.max_page_size)
                                .next_multiple_of(alignment),
                        },
                        dealloc_period: None,
                    },
                ]
            }
            MemoryConfiguration::ExclusivePages => {
                // Add all bin sizes. Nb: because of alignment some buckets
                // end up as the same size, so only want unique ones,
                // but also keep the order, so a BTree will do.
                const MIN_BUCKET_SIZE: u64 = 1024 * 32;
                const NUM_POOLS: usize = 24;

                let sizes = generate_bucket_sizes(
                    MIN_BUCKET_SIZE,
                    properties.max_page_size,
                    NUM_POOLS,
                    properties.alignment,
                );

                sizes
                    .iter()
                    .map(|&size| {
                        let dealloc_period = (BASE_DEALLOC_PERIOD as f64
                            * (1.0 + size as f64 / (DEALLOC_SCALE_MB as f64)).round())
                            as u64;

                        MemoryPoolOptions {
                            pool_type: PoolType::ExclusivePages {
                                max_alloc_size: size,
                            },
                            dealloc_period: Some(dealloc_period),
                        }
                    })
                    .collect()
            }
        }
    }
}

fn generate_bucket_sizes(
    start_size: u64,
    end_size: u64,
    max_buckets: usize,
    alignment: u64,
) -> Vec<u64> {
    let mut buckets = Vec::with_capacity(max_buckets);
    let log_min = (start_size as f64).ln();
    let log_max = (end_size as f64).ln();
    let log_range = log_max - log_min;

    // Pure exponential performed best, but let's try slightly denser in lower-mid range
    for i in 0..max_buckets {
        let p = i as f64 / (max_buckets - 1) as f64;
        // Slight bias toward lower-mid range with less aggressive curve than sigmoid
        let log_size = log_min + log_range * p;
        let size = log_size.exp() as u64;
        let aligned_size = size.next_multiple_of(alignment);
        buckets.push(aligned_size);
    }

    buckets.dedup();
    buckets
}

/// The `Adaptive` preset's small-allocation pool: its page size, and the
/// largest allocation routed to it.
#[cfg(not(exclusive_memory_only))]
const ADAPTIVE_SMALL_PAGE: u64 = 8 * 1024 * 1024;
#[cfg(not(exclusive_memory_only))]
const ADAPTIVE_SMALL_SLICE: u64 = 64 * 1024;
/// The `Adaptive` preset's smallest adaptive page, capped by the device's
/// `max_page_size`: what the smallest allocation it serves (just past
/// [`ADAPTIVE_SMALL_SLICE`]) needs anyway once rounded, so a stream that only
/// makes small allocations holds a page its size rather than a floor's.
#[cfg(not(exclusive_memory_only))]
const ADAPTIVE_MIN_PAGE: u64 = 2 * 1024 * 1024;

const DEALLOC_SCALE_MB: u64 = 1024 * 1024 * 1024;
const BASE_DEALLOC_PERIOD: u64 = 5000;
