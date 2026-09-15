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
            MemoryConfiguration::SubSlices => {
                // Round chunk size to be aligned.
                let memory_alignment = properties.alignment;
                let max_page = properties.max_page_size;
                let mut pools = Vec::new();

                const MB: u64 = 1024 * 1024;

                // Add in a pool for allocations that are smaller than the min alignment,
                // as they can't use offsets at all (on wgpu at least).
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages { max_alloc_size: 0 },
                    dealloc_period: None,
                });

                let mut current = max_page;
                let mut max_sizes = vec![];
                let mut page_sizes = vec![];
                let mut base = pools.len() as u32;

                while current >= 32 * MB {
                    current /= 4;

                    // Make sure every pool has an aligned size.
                    current = current.next_multiple_of(memory_alignment);

                    max_sizes.push(current / 2u64.pow(base));
                    page_sizes.push(current);
                    base += 1;
                }

                max_sizes.reverse();
                page_sizes.reverse();

                for i in 0..max_sizes.len() {
                    let max = max_sizes[i];
                    let page_size = page_sizes[i];

                    pools.push(MemoryPoolOptions {
                        // Creating max slices lower than the chunk size reduces fragmentation.
                        pool_type: PoolType::SlicedPages {
                            page_size,
                            max_slice_size: max,
                            max_pool_size: None,
                        },
                        dealloc_period: None,
                    });
                }

                // Allocations bigger than the sliced ladder get exact-size
                // exclusive pages. A sliced tail pool here would materialize a
                // whole `max_page` page (a quarter of device memory) for the
                // first allocation that lands in it — on unified-memory devices
                // that alone can consume a large share of host RAM. Exclusive
                // pages allocate exactly what is requested and are released once
                // they sit unused for a full dealloc period.
                let max_alloc = max_page / memory_alignment * memory_alignment;
                let dealloc_period = (BASE_DEALLOC_PERIOD as f64
                    * (1.0 + max_alloc as f64 / (DEALLOC_SCALE_MB as f64)).round())
                    as u64;
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: max_alloc,
                    },
                    dealloc_period: Some(dealloc_period),
                });
                pools
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
            MemoryConfiguration::Custom { pool_options } => pool_options,
        }
    }

    /// Whether this is the [`SubSlices`](Self::SubSlices) preset — never, in
    /// a build that has none.
    pub fn is_sub_slices(&self) -> bool {
        match self {
            #[cfg(not(exclusive_memory_only))]
            Self::SubSlices => true,
            _ => false,
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

const DEALLOC_SCALE_MB: u64 = 1024 * 1024 * 1024;
const BASE_DEALLOC_PERIOD: u64 = 5000;
