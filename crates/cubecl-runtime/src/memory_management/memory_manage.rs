use super::{
    memory_pool::{ExclusiveMemoryPool, MemoryPool, SliceBinding, SliceHandle, SlicedPool},
    MemoryConfiguration, MemoryDeviceProperties, MemoryPoolOptions, MemoryUsage, PoolType,
};
use crate::storage::{ComputeStorage, StorageHandle};
use alloc::vec::Vec;

enum DynamicPool {
    Sliced(SlicedPool),
    Exclusive(ExclusiveMemoryPool),
}

const MB: usize = 1024 * 1024;

impl MemoryPool for DynamicPool {
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        match self {
            DynamicPool::Sliced(m) => m.get(binding),
            DynamicPool::Exclusive(m) => m.get(binding),
        }
    }

    fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> SliceHandle {
        match self {
            DynamicPool::Sliced(m) => m.reserve(storage, size),
            DynamicPool::Exclusive(m) => m.reserve(storage, size),
        }
    }

    fn alloc<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: u64) -> SliceHandle {
        match self {
            DynamicPool::Sliced(m) => m.alloc(storage, size),
            DynamicPool::Exclusive(m) => m.alloc(storage, size),
        }
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicPool::Sliced(m) => m.get_memory_usage(),
            DynamicPool::Exclusive(m) => m.get_memory_usage(),
        }
    }

    fn max_alloc_size(&self) -> u64 {
        match self {
            DynamicPool::Sliced(m) => m.max_alloc_size(),
            DynamicPool::Exclusive(m) => m.max_alloc_size(),
        }
    }
    fn cleanup<Storage: ComputeStorage>(&mut self, storage: &mut Storage, alloc_nr: u64) {
        match self {
            DynamicPool::Sliced(m) => m.cleanup(storage, alloc_nr),
            DynamicPool::Exclusive(m) => m.cleanup(storage, alloc_nr),
        }
    }
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct MemoryManagement<Storage> {
    pools: Vec<DynamicPool>,
    storage: Storage,
    alloc_reserve_count: u64,
}

fn generate_bucket_sizes(
    min_alloc_size: u64,
    max_alloc_size: u64,
    max_buckets: usize,
    alignment: u64,
) -> Vec<u64> {
    let mut buckets = Vec::with_capacity(max_buckets);
    let log_min = (min_alloc_size as f64).ln();
    let log_max = (max_alloc_size as f64).ln();
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

impl<Storage: ComputeStorage> MemoryManagement<Storage> {
    /// Creates the options from device limits.
    pub fn from_configuration(
        storage: Storage,
        properties: MemoryDeviceProperties,
        config: MemoryConfiguration,
    ) -> Self {
        let pools = match config {
            #[cfg(not(exclusive_memory_only))]
            MemoryConfiguration::SubSlices => {
                // Round chunk size to be aligned.
                let memory_alignment = properties.alignment;
                let max_page = properties.max_page_size;

                let mut pools = Vec::new();
                pools.push(MemoryPoolOptions {
                    page_size: max_page / memory_alignment * memory_alignment, // align the size to max_page.
                    chunk_num_prealloc: 0,
                    pool_type: PoolType::SlicedPages {
                        max_slice_size: max_page,
                    },
                    dealloc_period: None,
                });

                const MB: u64 = 1024 * 1024;

                let mut current = max_page;
                while current >= 32 * MB {
                    current /= 4;
                    // Make sure every pool has an aligned size.
                    current = current.next_multiple_of(memory_alignment);

                    pools.push(MemoryPoolOptions {
                        page_size: current,
                        chunk_num_prealloc: 0,
                        // Creating max slices lower than the chunk size reduces fragmentation.
                        pool_type: PoolType::SlicedPages {
                            max_slice_size: current / 2u64.pow(pools.len() as u32),
                        },
                        dealloc_period: None,
                    });
                }
                // Add in a pool for allocations that are smaller than the min alignment,
                // as they can't use offsets at all (on wgpu at least).
                pools.push(MemoryPoolOptions {
                    page_size: memory_alignment,
                    chunk_num_prealloc: 0,
                    pool_type: PoolType::ExclusivePages,
                    dealloc_period: None,
                });
                pools
            }
            MemoryConfiguration::ExclusivePages => {
                // Add all bin sizes. Nb: because of alignment some buckets
                // end up as the same size, so only want unique ones,
                // but also keep the order, so a BTree will do.
                let sizes = generate_bucket_sizes(
                    8192,
                    properties.max_page_size,
                    256,
                    properties.alignment,
                );

                // Add in one pool for all massive allocations.
                sizes
                    .iter()
                    .map(|&s| {
                        // Bigger buckets will logically have less slices, and are a bigger win
                        // to deallocate, so make the deallocation period roughly proportional to
                        // alloc size.
                        //
                        // This also +- follows zipfs law https://en.wikipedia.org/wiki/Zipf%27s_law
                        // which is an ok assumption for the distribution of allocations.
                        //
                        // This ranges from:
                        //   128 bytes, 8389608 allocations (aka almost never)
                        //   10kb, 105857 allocations
                        //   1MB, 2024 allocations
                        //   100MB+, 1000-1011 allocations
                        let base_period = 250;
                        let dealloc_period = base_period + 128 * MB as u64 / (s + 1024);

                        MemoryPoolOptions {
                            page_size: s,
                            chunk_num_prealloc: 0,
                            pool_type: PoolType::ExclusivePages,
                            dealloc_period: Some(dealloc_period),
                        }
                    })
                    .collect()
            }
            MemoryConfiguration::Custom(pool_settings) => pool_settings,
        };

        for pool in pools.iter() {
            log::trace!("Using memory pool: \n {pool:?}");
        }

        Self::new(storage, pools, properties.alignment)
    }

    /// Creates a new instance using the given storage, merging_strategy strategy and slice strategy.
    pub fn new(mut storage: Storage, pools: Vec<MemoryPoolOptions>, memory_alignment: u64) -> Self {
        let mut pools: Vec<_> = pools
            .iter()
            .map(|options| {
                let mut pool = match options.pool_type {
                    PoolType::SlicedPages {
                        max_slice_size: max_slice,
                    } => DynamicPool::Sliced(SlicedPool::new(
                        options.page_size,
                        max_slice,
                        memory_alignment,
                    )),
                    PoolType::ExclusivePages => DynamicPool::Exclusive(ExclusiveMemoryPool::new(
                        options.page_size,
                        memory_alignment,
                        options.dealloc_period.unwrap_or(u64::MAX),
                    )),
                };

                for _ in 0..options.chunk_num_prealloc {
                    pool.alloc(&mut storage, options.page_size);
                }

                pool
            })
            .collect();

        pools.sort_by(|pool1, pool2| u64::cmp(&pool1.max_alloc_size(), &pool2.max_alloc_size()));

        Self {
            pools,
            storage,
            alloc_reserve_count: 0,
        }
    }

    /// Cleanup allocations in pools that are deemed unnecessary.
    pub fn cleanup(&mut self) {
        for pool in self.pools.iter_mut() {
            pool.cleanup(&mut self.storage, self.alloc_reserve_count);
        }
    }

    /// Returns the storage from the specified binding
    pub fn get(&mut self, binding: SliceBinding) -> Option<StorageHandle> {
        self.pools.iter().find_map(|p| p.get(&binding)).cloned()
    }

    /// Returns the resource from the storage at the specified handle
    pub fn get_resource(
        &mut self,
        binding: SliceBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Option<Storage::Resource> {
        let handle = self.get(binding);

        handle.map(|handle| {
            let handle = match offset_start {
                Some(offset) => handle.offset_start(offset),
                None => handle,
            };
            let handle = match offset_end {
                Some(offset) => handle.offset_end(offset),
                None => handle,
            };
            self.storage().get(&handle)
        })
    }

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    pub fn reserve(&mut self, size: u64) -> SliceHandle {
        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.alloc_reserve_count += 1;

        // Find first pool where size <= p.max_alloc with a binary search.
        let pool_ind = self.pools.partition_point(|p| size > p.max_alloc_size());

        // Ensure the pool index is in bounds, otherwise there isn't any pool that can fit the
        // requested allocation
        if pool_ind >= self.pools.len() {
            panic!("Unable to find valid pool partition point: No memory pool big enough to reserve {size} bytes.");
        }

        let pool = &mut self.pools[pool_ind];
        if pool.max_alloc_size() < size {
            panic!("No memory pool big enough to reserve {size} bytes.");
        }
        pool.reserve(&mut self.storage, size)
    }

    /// Bypass the memory allocation algorithm to allocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    pub fn alloc(&mut self, size: u64) -> SliceHandle {
        // Find first pool where size <= p.max_alloc with a binary search.
        let pool_ind = self.pools.partition_point(|p| size > p.max_alloc_size());
        let pool = &mut self.pools[pool_ind];
        if pool.max_alloc_size() < size {
            panic!("No memory pool big enough to alloc {size} bytes.");
        }
        pool.alloc(&mut self.storage, size)
    }

    /// Bypass the memory allocation algorithm to deallocate data directly.
    ///
    /// # Notes
    ///
    /// Can be useful for servers that want specific control over memory.
    pub fn dealloc(&mut self, _binding: SliceBinding) {
        // Can't dealloc slices.
    }

    /// Fetch the storage used by the memory manager.
    ///
    /// # Notes
    ///
    /// The storage should probably not be used for allocations since the handles won't be
    /// compatible with the ones provided by the current trait. Prefer using the
    /// [alloc](MemoryManagement::alloc) and [dealloc](MemoryManagement::dealloc) functions.
    ///
    /// This is useful if you need to time the deallocations based on async computation, or to
    /// change the mode of storage for different reasons.
    pub fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }

    /// Get the current memory usage.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.pools.iter().map(|x| x.get_memory_usage()).fold(
            MemoryUsage {
                number_allocs: 0,
                bytes_in_use: 0,
                bytes_padding: 0,
                bytes_reserved: 0,
            },
            |m1, m2| m1.combine(m2),
        )
    }

    /// Print out a report of the current memory usage.
    pub fn print_memory_usage(&self) {
        #[cfg(feature = "std")]
        log::info!("{}", self.memory_usage());
    }
}

impl<Storage> core::fmt::Debug for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "DynamicMemoryManagement {:?}",
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory_management::MemoryManagement, storage::BytesStorage};

    // Test pools with slices.
    #[test]
    #[cfg(not(exclusive_memory_only))]
    fn test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
        );
        let handle = memory_management.reserve(10);
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test]
    fn alloc_two_chunks_on_one_page() {
        let page_size = 2048;

        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::SlicedPages {
                    max_slice_size: page_size,
                },
                dealloc_period: None,
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test]
    fn alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let page_size = 512;

        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::SlicedPages {
                    max_slice_size: page_size,
                },
                dealloc_period: None,
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test]
    fn alloc_allocs_new_storage() {
        let page_size = 1024;

        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::SlicedPages {
                    max_slice_size: page_size,
                },
                dealloc_period: None,
            }],
            32,
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test]
    fn alloc_respects_alignment_size() {
        let page_size = 500;
        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::SlicedPages {
                    max_slice_size: page_size,
                },
                dealloc_period: None,
            }],
            50,
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 60 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test]
    fn allocs_on_correct_page() {
        let sizes = [100, 200, 300, 400];

        let pools = sizes
            .iter()
            .map(|&size| MemoryPoolOptions {
                page_size: size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::SlicedPages {
                    max_slice_size: size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::new(BytesStorage::default(), pools, 10);
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));

        let usage = memory_management.memory_usage();

        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
        assert_eq!(usage.bytes_reserved, sizes.iter().sum::<u64>());
    }

    #[test]
    #[cfg(not(exclusive_memory_only))]
    fn allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        // Usage after can actually be _less_ because of defragging.
        assert!(usage_before.bytes_reserved >= usage_after.bytes_reserved);
    }

    #[test]
    #[cfg(not(exclusive_memory_only))]
    fn test_fragmentation_resistance() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
        );
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| memory_management.reserve(size))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management.reserve(size);
        }
        let usage_after = memory_management.memory_usage();
        // Check that we haven't increased our memory usage significantly
        assert!(usage_after.bytes_reserved <= (usage_before.bytes_reserved as f64 * 1.1) as u64);
    }

    // Test pools without slices. More or less same as tests above.
    #[test]
    fn noslice_test_handle_mutability() {
        let mem_props = MemoryDeviceProperties {
            max_page_size: 128 * 1024 * 1024,
            alignment: 32,
        };
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            mem_props,
            MemoryConfiguration::ExclusivePages,
        );
        let handle = memory_management.reserve(10);
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test]
    fn noslice_alloc_two_chunk() {
        let page_size = 2048;

        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::ExclusivePages,
                dealloc_period: None,
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test]
    fn noslice_alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size: 512,
                chunk_num_prealloc: 0,
                pool_type: PoolType::ExclusivePages,
                dealloc_period: None,
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, alloc_size);
    }

    #[test]
    fn noslice_alloc_allocs_new_storage() {
        let page_size = 1024;
        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::ExclusivePages,
                dealloc_period: None,
            }],
            32,
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test]
    fn noslice_alloc_respects_alignment_size() {
        let page_size = 500;
        let mut memory_management = MemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::ExclusivePages,
                dealloc_period: None,
            }],
            50,
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 60 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test]
    fn noslice_allocs_on_correct_page() {
        let pools = [100, 200, 300, 400]
            .iter()
            .map(|&size| MemoryPoolOptions {
                page_size: size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::SlicedPages {
                    max_slice_size: size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::new(BytesStorage::default(), pools, 10);
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));
        let usage = memory_management.memory_usage();
        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
    }

    #[test]
    fn noslice_allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::ExclusivePages,
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }
}
