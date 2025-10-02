use super::{
    MemoryConfiguration, MemoryDeviceProperties, MemoryPoolOptions, MemoryUsage, PoolType,
    memory_pool::{ExclusiveMemoryPool, MemoryPool, SlicedPool, StaticPool, VirtualMemoryPool},
};
use crate::{
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};

#[cfg(not(exclusive_memory_only))]
use alloc::vec;
use alloc::vec::Vec;

pub use super::memory_pool::{SliceBinding, handle::*};

// These are 288 bytes vs 64 bytes. Adding boxing isn't really worth
// saving the 200 bytes.
#[allow(clippy::large_enum_variant)]
enum DynamicPool {
    Sliced(SlicedPool),
    Exclusive(ExclusiveMemoryPool),
    Virtual(VirtualMemoryPool),
}

impl MemoryPool for DynamicPool {
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        match self {
            DynamicPool::Sliced(m) => m.get(binding),
            DynamicPool::Exclusive(m) => m.get(binding),
            DynamicPool::Virtual(m) => m.get(binding),
        }
    }

    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        match self {
            DynamicPool::Sliced(m) => m.try_reserve(size),
            DynamicPool::Exclusive(m) => m.try_reserve(size),
            DynamicPool::Virtual(m) => m.try_reserve(size),
        }
    }

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.alloc(storage, size),
            DynamicPool::Exclusive(m) => m.alloc(storage, size),
            DynamicPool::Virtual(m) => m.alloc(storage, size),
        }
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicPool::Sliced(m) => m.get_memory_usage(),
            DynamicPool::Exclusive(m) => m.get_memory_usage(),
            DynamicPool::Virtual(m) => m.get_memory_usage(),
        }
    }

    fn max_alloc_size(&self) -> u64 {
        match self {
            DynamicPool::Sliced(m) => m.max_alloc_size(),
            DynamicPool::Exclusive(m) => m.max_alloc_size(),
            DynamicPool::Virtual(m) => m.max_alloc_size(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        match self {
            DynamicPool::Sliced(m) => m.cleanup(storage, alloc_nr, explicit),
            DynamicPool::Exclusive(m) => m.cleanup(storage, alloc_nr, explicit),
            DynamicPool::Virtual(m) => m.cleanup(storage, alloc_nr, explicit),
        }
    }
}

#[derive(Default, Clone, Copy)]
/// The mode of allocation used.
pub enum MemoryAllocationMode {
    /// Use the automatic memory management strategy for allocation.
    #[default]
    Auto,
    /// Use a static memory management strategy, meaning that all allocations are for data that is
    /// never going to be freed.
    Static,
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct MemoryManagement<Storage> {
    static_pool: StaticPool,
    pools: Vec<DynamicPool>,
    storage: Storage,
    alloc_reserve_count: u64,
    mode: MemoryAllocationMode,
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

impl<Storage: ComputeStorage> MemoryManagement<Storage> {
    /// Creates the options from device limits.
    pub fn from_configuration(
        storage: Storage, // It is needed to make this dynamically dispatch at runtime to be compatible with virtualstorage and pool.
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
    ) -> Self {
        let pool_options = match config {
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
                        },
                        dealloc_period: None,
                    });
                }

                // Add pools from big to small.
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size: max_page / memory_alignment * memory_alignment,
                        max_slice_size: max_page / memory_alignment * memory_alignment,
                    },
                    dealloc_period: None,
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
            MemoryConfiguration::VirtualMemory => {
                if properties.virtual_memory_supported && storage.is_virtual_mem_enabled() {
                    // Collect the minimum granularity of the target device.
                    let minimum_granularity = properties.min_granularity;
                    // I am following `PyTorch approach here. PyTorch maintains two virtual memory pools.
                    // One of them makes allocations aligned to 10 * min_granularity (huge pages), mainly useful with LLMs and very large models that allocate huge tensors.

                    let big_alloc_size = 10 * minimum_granularity; // Around 20MiB
                    let small_alloc_size = minimum_granularity; // Around 2MiB

                    // The other makes finer grained allocations aligned to gpu page size. It is better to set the second one's max_alloc_size equal to the granularity, as it will make memory reuse easier by enforcing all physical allocations to be the same size.
                    let small_dealloc_period = (BASE_DEALLOC_PERIOD as f64
                        * (1.0 + small_alloc_size as f64 / (DEALLOC_SCALE_MB as f64)).round())
                        as u64;

                    let big_dealloc_period = (BASE_DEALLOC_PERIOD as f64
                        * (1.0 + big_alloc_size as f64 / (DEALLOC_SCALE_MB as f64)).round())
                        as u64;
                    vec![
                        // Smaller pool, fixed size physical allocations,
                        // Always allocates memory of about 2MiB.
                        MemoryPoolOptions {
                            pool_type: PoolType::VirtualMemory {
                                alloc_size: small_alloc_size,
                            },
                            dealloc_period: Some(small_dealloc_period),
                        },
                        // Bigger pool.
                        // Max alloc size is 10 * min_granularity (about 20MiB)
                        // Alignment is half that value, so the pool will allocate physical blocks of sizes ranging from 10MiB to 20MiB.
                        MemoryPoolOptions {
                            pool_type: PoolType::VirtualMemory {
                                alloc_size: big_alloc_size,
                            },
                            // The dealloc period depends on the expected size of the allocations.
                            dealloc_period: Some(big_dealloc_period),
                        },
                    ]
                // In case virtual memory is not set to be supported return an empty vector, which in practice will fallback to use the static pool.
                } else {
                    Vec::new()
                }
            }
            MemoryConfiguration::Custom { pool_options } => pool_options,
        };

        for pool in pool_options.iter() {
            log::trace!("Using memory pool: \n {pool:?}");
        }

        let pools: Vec<_> = pool_options
            .iter()
            .map(|options| match options.pool_type {
                PoolType::SlicedPages {
                    page_size,
                    max_slice_size,
                } => DynamicPool::Sliced(SlicedPool::new(
                    page_size,
                    max_slice_size,
                    properties.alignment,
                )),
                PoolType::ExclusivePages { max_alloc_size } => {
                    DynamicPool::Exclusive(ExclusiveMemoryPool::new(
                        max_alloc_size,
                        properties.alignment,
                        options.dealloc_period.unwrap_or(u64::MAX),
                    ))
                }
                PoolType::VirtualMemory { alloc_size } => {
                    DynamicPool::Virtual(VirtualMemoryPool::new(
                        alloc_size,
                        properties.min_granularity,
                        options.dealloc_period.unwrap_or(u64::MAX),
                    ))
                }
            })
            .collect();

        Self {
            static_pool: StaticPool::new(properties.max_page_size),
            pools,
            storage,
            alloc_reserve_count: 0,
            mode: MemoryAllocationMode::Auto,
        }
    }

    /// Change the mode of allocation.
    pub fn mode(&mut self, mode: MemoryAllocationMode) {
        self.mode = mode;
    }

    /// Cleanup allocations in pools that are deemed unnecessary.
    pub fn cleanup(&mut self, explicit: bool) {
        self.static_pool
            .cleanup(&mut self.storage, self.alloc_reserve_count, explicit);

        for pool in self.pools.iter_mut() {
            pool.cleanup(&mut self.storage, self.alloc_reserve_count, explicit);
        }
    }

    /// Returns the storage from the specified binding
    pub fn get(&mut self, binding: SliceBinding) -> Option<StorageHandle> {
        if let Some(val) = self.static_pool.get(&binding) {
            return Some(val.clone());
        }

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
    pub fn reserve(&mut self, size: u64) -> Result<SliceHandle, IoError> {
        if let MemoryAllocationMode::Static = self.mode {
            return self.static_pool.alloc(&mut self.storage, size);
        }

        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.alloc_reserve_count += 1;

        // Find first pool that fits this allocation
        let pool = self
            .pools
            .iter_mut()
            .find(|p| p.max_alloc_size() >= size)
            .ok_or(IoError::BufferTooBig(size as usize))?;

        if let Some(slice) = pool.try_reserve(size) {
            return Ok(slice);
        }

        pool.alloc(&mut self.storage, size)
    }

    /// Fetch the storage used by the memory manager.
    ///
    /// # Notes
    ///
    /// The storage should probably not be used for allocations since the handles won't be
    /// compatible with the ones provided by the current trait. Prefer using the
    /// [alloc](ComputeStorage::alloc) and [dealloc](ComputeStorage::dealloc) functions.
    ///
    /// This is useful if you need to time the deallocations based on async computation, or to
    /// change the mode of storage for different reasons.
    pub fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }

    /// Get the current memory usage.
    pub fn memory_usage(&self) -> MemoryUsage {
        let memory_usage = self.pools.iter().map(|x| x.get_memory_usage()).fold(
            MemoryUsage {
                number_allocs: 0,
                bytes_in_use: 0,
                bytes_padding: 0,
                bytes_reserved: 0,
            },
            |m1, m2| m1.combine(m2),
        );
        memory_usage.combine(self.static_pool.get_memory_usage())
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
    use crate::storage::BytesAllocationMode;
    use crate::{
        memory_management::MemoryManagement,
        storage::{BytesStorage, BytesVirtualStorage},
    };

    const DUMMY_MEM_PROPS: MemoryDeviceProperties = MemoryDeviceProperties {
        max_page_size: 128 * 1024 * 1024,
        alignment: 32,
        data_transfer_async: false,
        virtual_memory_supported: false,
        min_granularity: 0,
    };

    const VIRTUAL_MEM_PROPS: MemoryDeviceProperties = MemoryDeviceProperties {
        max_page_size: 64,
        alignment: 64,
        data_transfer_async: false,
        virtual_memory_supported: true,
        min_granularity: 64,
    };

    fn create_virtual_memory_manager() -> MemoryManagement<BytesVirtualStorage> {
        let storage = BytesVirtualStorage::new(
            VIRTUAL_MEM_PROPS.alignment as usize,
            BytesAllocationMode::Virtual,
        );
        MemoryManagement::from_configuration(
            storage,
            &VIRTUAL_MEM_PROPS,
            MemoryConfiguration::VirtualMemory,
        )
    }

    // Test pools with slices.
    #[test]
    #[cfg(not(exclusive_memory_only))]
    fn test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::SubSlices,
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    // Test pools with slices.
    #[test]
    #[cfg(not(exclusive_memory_only))]
    fn test_memory_usage() {
        let max_page_size = 512;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: max_page_size,
                    },
                    dealloc_period: None,
                }],
            },
        );
        let handle = memory_management.reserve(100);
        let usage = memory_management.memory_usage();

        assert_eq!(usage.bytes_in_use, 100);
        assert!(usage.bytes_reserved >= 100 && usage.bytes_reserved <= max_page_size);

        // Drop and re-alloc.
        drop(handle);
        let _handle = memory_management.reserve(100);
        let usage_new = memory_management.memory_usage();
        assert_eq!(usage, usage_new);
    }

    #[test]
    fn alloc_two_chunks_on_one_page() {
        let page_size = 2048;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
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

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
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

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
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
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: page_size,
                alignment: 50,
                data_transfer_async: false,
                virtual_memory_supported: true,
                min_granularity: 0,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 50 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test]
    fn allocs_on_correct_page() {
        let sizes = [100, 200, 300, 400];

        let pools = sizes
            .iter()
            .map(|size| MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: *size,
                    max_slice_size: *size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 10,
                data_transfer_async: false,
                virtual_memory_supported: false,
                min_granularity: 0,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));

        let usage = memory_management.memory_usage();

        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
        assert!(usage.bytes_reserved >= sizes.iter().sum::<u64>());
    }

    #[test]
    #[cfg(not(exclusive_memory_only))]
    fn allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
                data_transfer_async: false,
                virtual_memory_supported: false,
                min_granularity: 0,
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
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
                data_transfer_async: false,
                virtual_memory_supported: false,
                min_granularity: 0,
            },
            MemoryConfiguration::SubSlices,
        );
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| memory_management.reserve(size).unwrap())
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management.reserve(size).unwrap();
        }
        let usage_after = memory_management.memory_usage();
        // Check that we haven't increased our memory usage significantly
        assert!(usage_after.bytes_reserved <= (usage_before.bytes_reserved as f64 * 1.1) as u64);
    }

    // Test pools without slices. More or less same as tests above.
    #[test]
    fn noslice_test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &(MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
                data_transfer_async: false,
                virtual_memory_supported: false,
                min_granularity: 0,
            }),
            MemoryConfiguration::ExclusivePages,
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test]
    fn noslice_alloc_two_chunk() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test]
    fn noslice_alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert!(usage.bytes_reserved >= alloc_size);
    }

    #[test]
    fn noslice_alloc_allocs_new_storage() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test]
    fn noslice_alloc_respects_alignment_size() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 50,
                data_transfer_async: false,
                virtual_memory_supported: false,
                min_granularity: 0,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 50 * 20,
                    },
                    dealloc_period: None,
                }],
            },
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
                pool_type: PoolType::SlicedPages {
                    page_size: size,
                    max_slice_size: size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 10,
                data_transfer_async: false,
                virtual_memory_supported: false,
                min_granularity: 0,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
        );
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
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
                data_transfer_async: false,
                virtual_memory_supported: false,
                min_granularity: 0,
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

    #[test]
    // Validates that we can safely perform allocations on the virtual pool,
    // And that data integrity is respected.
    fn test_virtual_pool_alloc() {
        // Tests the overall functionality of the virtual memory pool.
        let mut memory_manager = create_virtual_memory_manager();
        let page_size = 64 * 2; // Small size to go to smaller pool

        // Allocate multiple blocks that will create separate pages
        let mut handles = Vec::with_capacity(4);
        for i in 0..4 {
            let handle = memory_manager.reserve(page_size).unwrap();
            let resource = memory_manager
                .get_resource(handle.clone().binding(), None, None)
                .unwrap();
            resource.write()[0] = (i + 10) as u8; // Unique marker to be able to later check that data integrity is preserved.
            handles.push(handle);
        }

        let (num_pages, num_slices) = match &mut memory_manager.pools[0] {
            DynamicPool::Virtual(p) => (p.pages.len(), p.virtual_slices.len()),
            _ => panic!("Should be using Virtual Memory Pool"),
        };

        assert!(num_pages == 4); // Four pages should have been allocated.
        assert!(num_slices == 4); // Four slices should have been allocated.

        // Validate data integrity
        for (i, handle) in handles.iter().enumerate() {
            let resource = memory_manager
                .get_resource(handle.clone().binding(), None, None)
                .unwrap();
            assert_eq!(resource.read()[0], (i + 10) as u8);
        }
    }

    #[test]
    /// Validates that defragmentation happens after explicitly calling cleanup
    fn test_virtual_pool_cleanup() {
        // Tests the overall functionality of the virtual memory pool.
        let mut memory_manager = create_virtual_memory_manager();
        let page_size = 64 * 2; // Small size to go to smaller pool
        {
            // Allocate multiple blocks that will create separate pages
            let mut handles = Vec::with_capacity(4);
            for i in 0..4 {
                let handle = memory_manager.reserve(page_size).unwrap();
                let resource = memory_manager
                    .get_resource(handle.clone().binding(), None, None)
                    .unwrap();
                resource.write()[0] = (i + 10) as u8; // Unique marker to be able to later check that data integrity is preserved.
                handles.push(handle);
            }

            let (num_pages, num_slices) = match &mut memory_manager.pools[0] {
                DynamicPool::Virtual(p) => (p.pages.len(), p.virtual_slices.len()),
                _ => panic!("Should be using Virtual Memory Pool"),
            };

            assert!(num_pages == 4); // Four pages should have been allocated.
            assert!(num_slices == 4); // Four slices should have been allocated.

            // Validate data integrity
            for (i, handle) in handles.iter().enumerate() {
                let resource = memory_manager
                    .get_resource(handle.clone().binding(), None, None)
                    .unwrap();
                assert_eq!(resource.read()[0], (i + 10) as u8);
            }
        }

        memory_manager.cleanup(true);

        let usage = memory_manager.memory_usage();

        // Bytes in use should be zero as memory has been deallocated.
        assert_eq!(usage.bytes_in_use, 0);
        // Bytes reserved should be zero, as we have called it explicitly and therefore no free physical memory should be kept.
        assert_eq!(usage.bytes_reserved, 0);

        // Check if merging has worked.
        let (num_pages, num_slices) = match &mut memory_manager.pools[0] {
            DynamicPool::Virtual(p) => (p.pages.len(), p.virtual_slices.len()),
            _ => panic!("Should be using Virtual Memory Pool"),
        };

        assert_eq!(num_pages, 1); // Pages should have been merged in a single one.
        assert_eq!(num_slices, 1); // Slices should have been merged after defragmentation
    }

    #[test]
    // This pattern is more complex to better test defragmentation
    // Allocate foiur handles, deallocate two of them, defragment.
    // The final page count and slice count should be three as two of the pages are currently allocated and cannot be merged.
    fn test_virtual_pool_cleanup_no_explicit() {
        // Tests the overall functionality of the virtual memory pool.
        let mut memory_manager = create_virtual_memory_manager();
        let page_size = 64 * 2; // Small size to go to smaller pool

        // Allocate multiple blocks that will create separate pages
        let mut handles = Vec::with_capacity(4);

        for i in 0..4 {
            let handle = memory_manager.reserve(page_size).unwrap();
            let resource = memory_manager
                .get_resource(handle.clone().binding(), None, None)
                .unwrap();
            resource.write()[0] = (i + 10) as u8; // Unique marker to be able to later check that data integrity is preserved.

            handles.push(handle);
        }

        let dealloc_period = match &mut memory_manager.pools[0] {
            DynamicPool::Virtual(p) => {
                p.reset_tracking();
                p.dealloc_period
            }
            _ => panic!("Should be using Virtual Memory Pool"),
        };
        // Clear the recently allocated vector.
        memory_manager.alloc_reserve_count = dealloc_period; //  force alloc reserve count to be dealloc period so that defragment is called at the following cleanup call

        let usage_before = memory_manager.memory_usage();

        // Extract handles to deallocate
        let handle_0 = handles.remove(0); // Remove index 0, others shift down
        let handle_2 = handles.remove(1); // What was index 2 is now index 1

        // Deallocate every other allocation to create fragmented pages
        drop(handle_0);
        drop(handle_2);

        let usage_after = memory_manager.memory_usage();

        assert!(usage_before.bytes_in_use == usage_after.bytes_in_use + 2 * page_size);

        // Cleanup without explicit call should return physical memory for reuse.
        memory_manager.cleanup(false);

        let available_mem = match &mut memory_manager.pools[0] {
            DynamicPool::Virtual(p) => p.reusable_memory(),
            _ => panic!("Should be using Virtual Memory Pool"),
        };

        assert!(available_mem > 0); // There should be some free handles ready to be reused
        let final_usage = memory_manager.memory_usage();

        assert!(final_usage.bytes_in_use == usage_after.bytes_in_use);
        assert!(final_usage.bytes_reserved == usage_after.bytes_reserved); // memory should not have been freed

        // Check if merging has worked.
        let (num_pages, num_slices) = match &mut memory_manager.pools[0] {
            DynamicPool::Virtual(p) => (p.pages.len(), p.virtual_slices.len()),
            _ => panic!("Should be using Virtual Memory Pool"),
        };

        assert_eq!(num_pages, 3); // Pages should have been merged in three, as we had four and have freed two, which should have merged in one
        assert_eq!(num_slices, 3); // Slices should have been merged after defragmentation.
    }
}
