use super::memory_pool::{
    MemoryPool, MemoryPoolOptions, MemoryUsage, PoolType, SimpleMemoryPool, SliceBinding,
    SliceHandle, SlicedPool,
};
use crate::storage::{ComputeStorage, StorageHandle, StorageId};
use alloc::vec::Vec;

use super::MemoryManagement;

enum DynamicPool {
    Sliced(SlicedPool),
    Simple(SimpleMemoryPool),
}

// Bin sizes as per https://github.com/sebbbi/OffsetAllocator/blob/main/README.md
// This guarantees that _for bins in use_, the wasted space is at most 12.5%. So as long
// as bins have a high use rate this should be fairly efficient. That said, currently slices in
// bins don't deallocate, so there is a chance more memory than needed is used.
const EXP_BIN_SIZES: [usize; 200] = [
    128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640,
    704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560,
    2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 9216,
    10240, 11264, 12288, 13312, 14336, 15360, 16384, 18432, 20480, 22528, 24576, 26624, 28672,
    30720, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536, 73728, 81920, 90112,
    98304, 106496, 114688, 122880, 131072, 147456, 163840, 180224, 196608, 212992, 229376, 245760,
    262144, 294912, 327680, 360448, 393216, 425984, 458752, 491520, 524288, 589824, 655360, 720896,
    786432, 851968, 917504, 983040, 1048576, 1179648, 1310720, 1441792, 1572864, 1703936, 1835008,
    1966080, 2097152, 2359296, 2621440, 2883584, 3145728, 3407872, 3670016, 3932160, 4194304,
    4718592, 5242880, 5767168, 6291456, 6815744, 7340032, 7864320, 8388608, 9437184, 10485760,
    11534336, 12582912, 13631488, 14680064, 15728640, 16777216, 18874368, 20971520, 23068672,
    25165824, 27262976, 29360128, 31457280, 33554432, 37748736, 41943040, 46137344, 50331648,
    54525952, 58720256, 62914560, 67108864, 75497472, 83886080, 92274688, 100663296, 109051904,
    117440512, 125829120, 134217728, 150994944, 167772160, 184549376, 201326592, 218103808,
    234881024, 251658240, 268435456, 301989888, 335544320, 369098752, 402653184, 436207616,
    469762048, 503316480, 536870912, 603979776, 671088640, 738197504, 805306368, 872415232,
    939524096, 1006632960, 1073741824, 1207959552, 1342177280, 1476395008, 1610612736, 1744830464,
    1879048192, 2013265920, 2147483648, 2415919104, 2684354560, 2952790016, 3221225472, 3489660928,
    3758096384, 4026531840,
];

impl MemoryPool for DynamicPool {
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        match self {
            DynamicPool::Sliced(m) => m.get(binding),
            DynamicPool::Simple(m) => m.get(binding),
        }
    }

    fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        exclude: &[StorageId],
    ) -> SliceHandle {
        match self {
            DynamicPool::Sliced(m) => m.reserve(storage, size, exclude),
            DynamicPool::Simple(m) => m.reserve(storage, size, exclude),
        }
    }

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> SliceHandle {
        match self {
            DynamicPool::Sliced(m) => m.alloc(storage, size),
            DynamicPool::Simple(m) => m.alloc(storage, size),
        }
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicPool::Sliced(m) => m.get_memory_usage(),
            DynamicPool::Simple(m) => m.get_memory_usage(),
        }
    }

    fn max_alloc_size(&self) -> usize {
        match self {
            DynamicPool::Sliced(m) => m.max_alloc_size(),
            DynamicPool::Simple(m) => m.max_alloc_size(),
        }
    }
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct DynamicMemoryManagement<Storage> {
    pools: Vec<DynamicPool>,
    storage: Storage,
}

/// High level configuration of memory management.
#[derive(Clone, Debug)]
pub enum MemoryConfiguration {
    /// Use the default preset.
    Default,
    /// Default preset for allocating _without_ any slices.
    WithoutSubSlices,
    /// Customize each pool individually.
    Custom(Vec<MemoryPoolOptions>),
}

/// Properties of the device related to allocation.
#[derive(Debug)]
pub struct MemoryDeviceProperties {
    /// The maximum nr. of bytes that can be allocated in one go.
    pub max_page_size: usize,
    /// The required memory offset alignment in bytes.
    pub memory_alignment: usize,
}

const MB: usize = 1024 * 1024;

impl<Storage: ComputeStorage> DynamicMemoryManagement<Storage> {
    /// Creates the options from device limits.
    pub fn from_configuration(
        storage: Storage,
        properties: MemoryDeviceProperties,
        config: MemoryConfiguration,
    ) -> Self {
        match config {
            MemoryConfiguration::Default => {
                // Round chunk size to be aligned.
                let memory_alignment = properties.memory_alignment;
                let max_page = (properties.max_page_size / memory_alignment) * memory_alignment;

                let mut pools = Vec::new();
                pools.push(MemoryPoolOptions {
                    page_size: max_page,
                    chunk_num_prealloc: 0,
                    pool_type: PoolType::Slices(max_page),
                });

                let mut current = max_page;
                while current >= 32 * MB {
                    current /= 4;
                    // Make sure every pool has an aligned size.
                    current = (current / memory_alignment) * memory_alignment;

                    pools.push(MemoryPoolOptions {
                        page_size: current,
                        chunk_num_prealloc: 0,
                        // Creating max slices lower than the chunk size reduces fragmentation.
                        pool_type: PoolType::Slices(current / 2usize.pow(pools.len() as u32)),
                    });
                }
                // Add in a pool for allocations that are smaller than the min alignment,
                // as they can't use offsets at all (on wgpu at least).
                pools.push(MemoryPoolOptions {
                    page_size: memory_alignment,
                    chunk_num_prealloc: 0,
                    pool_type: PoolType::NoSlices,
                });
                Self::new(storage, pools, memory_alignment)
            }
            MemoryConfiguration::WithoutSubSlices => {
                // Round chunk size to be aligned.
                let memory_alignment = properties.memory_alignment;
                // Use all bins up to max_page and add max_page as bin.
                let mut sizes: Vec<_> = EXP_BIN_SIZES
                    .iter()
                    .copied()
                    .map(|size| (size / memory_alignment) * memory_alignment)
                    .take_while(|&size| size < properties.max_page_size)
                    .collect();
                sizes.push((properties.max_page_size / memory_alignment) * memory_alignment);

                // Add in one pool for all massive allocations.
                let pools: Vec<_> = sizes
                    .iter()
                    .map(|&s| MemoryPoolOptions {
                        page_size: s,
                        chunk_num_prealloc: 0,
                        pool_type: PoolType::NoSlices,
                    })
                    .collect();

                Self::new(storage, pools, memory_alignment)
            }
            MemoryConfiguration::Custom(pool_settings) => {
                Self::new(storage, pool_settings, properties.memory_alignment)
            }
        }
    }

    /// Creates a new instance using the given storage, merging_strategy strategy and slice strategy.
    pub fn new(
        mut storage: Storage,
        pools: Vec<MemoryPoolOptions>,
        memory_alignment: usize,
    ) -> Self {
        let mut pools: Vec<_> = pools
            .iter()
            .map(|option| {
                let mut pool = match option.pool_type {
                    PoolType::Slices(max_slice) => DynamicPool::Sliced(SlicedPool::new(
                        option.page_size,
                        max_slice,
                        memory_alignment,
                    )),
                    PoolType::NoSlices => DynamicPool::Simple(SimpleMemoryPool::new(
                        option.page_size,
                        memory_alignment,
                    )),
                };

                for _ in 0..option.chunk_num_prealloc {
                    pool.alloc(&mut storage, option.page_size);
                }

                pool
            })
            .collect();

        pools.sort_by(|pool1, pool2| usize::cmp(&pool1.max_alloc_size(), &pool2.max_alloc_size()));

        Self { pools, storage }
    }
}

impl<Storage> core::fmt::Debug for DynamicMemoryManagement<Storage> {
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

impl<Storage: ComputeStorage> MemoryManagement<Storage> for DynamicMemoryManagement<Storage> {
    type Handle = SliceHandle;
    type Binding = SliceBinding;

    fn get(&mut self, binding: Self::Binding) -> StorageHandle {
        self.pools
            .iter()
            .find_map(|p| p.get(&binding))
            .expect("No handle found in memory pools")
            .clone()
    }

    fn reserve(&mut self, size: usize, exclude: &[StorageId]) -> Self::Handle {
        // Find first pool where size <= p.max_alloc with a binary search.
        let pool_ind = self.pools.partition_point(|p| size > p.max_alloc_size());
        let pool = &mut self.pools[pool_ind];
        if pool.max_alloc_size() < size {
            panic!("No memory pool big enough to reserve {size} bytes.");
        }
        pool.reserve(&mut self.storage, size, exclude)
    }

    fn alloc(&mut self, size: usize) -> Self::Handle {
        // Find first pool where size <= p.max_alloc with a binary search.
        let pool_ind = self.pools.partition_point(|p| size > p.max_alloc_size());
        let pool = &mut self.pools[pool_ind];
        if pool.max_alloc_size() < size {
            panic!("No memory pool big enough to alloc {size} bytes.");
        }
        pool.alloc(&mut self.storage, size)
    }

    fn dealloc(&mut self, _binding: Self::Binding) {
        // Can't dealloc slices.
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }
}

impl<Storage: ComputeStorage> DynamicMemoryManagement<Storage> {
    /// Get a report of the current memory usage.
    pub fn get_memory_usage(&self) -> MemoryUsage {
        self.pools
            .iter()
            .map(|x| x.get_memory_usage())
            .fold(MemoryUsage::default(), |m1, m2| m1.combine(m2))
    }

    /// Print out a report of the current memory usage.
    pub fn print_memory_usage(&self) {
        log::info!("{}", self.get_memory_usage());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory_management::MemoryManagement, storage::BytesStorage};

    // Test pools with slices.
    #[test]
    fn test_handle_mutability() {
        let mut memory_management = DynamicMemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                memory_alignment: 32,
            },
            MemoryConfiguration::Default,
        );
        let handle = memory_management.reserve(10, &[]);
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test]
    fn alloc_two_chunks_on_one_page() {
        let page_size = 2048;

        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::Slices(page_size),
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &[]);
        let _new_handle = memory_management.reserve(alloc_size, &[]);

        let usage = memory_management.get_memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test]
    fn alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let page_size = 512;

        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::Slices(page_size),
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &[]);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size, &[]);

        let usage = memory_management.get_memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test]
    fn alloc_allocs_new_storage() {
        let page_size = 1024;

        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::Slices(page_size),
            }],
            32,
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size, &[]);
        let _new_handle = memory_management.reserve(alloc_size, &[]);

        let usage = memory_management.get_memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test]
    fn alloc_respects_alignment_size() {
        let page_size = 500;
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::Slices(page_size),
            }],
            50,
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size, &[]);
        let _new_handle = memory_management.reserve(alloc_size, &[]);
        let usage = memory_management.get_memory_usage();
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
                pool_type: PoolType::Slices(size),
            })
            .collect();
        let mut memory_management =
            DynamicMemoryManagement::new(BytesStorage::default(), pools, 10);
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s, &[]));

        let usage = memory_management.get_memory_usage();

        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<usize>());
        assert_eq!(usage.bytes_reserved, sizes.iter().sum::<usize>());
    }

    #[test]
    fn allocate_deallocate_reallocate() {
        let mut memory_management = DynamicMemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                memory_alignment: 32,
            },
            MemoryConfiguration::Default,
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &[]))
            .collect();
        let usage_before = memory_management.get_memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &[]))
            .collect();
        let usage_after = memory_management.get_memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }

    #[test]
    fn test_fragmentation_resistance() {
        let mut memory_management = DynamicMemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                memory_alignment: 32,
            },
            MemoryConfiguration::Default,
        );
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| memory_management.reserve(size, &[]))
            .collect();
        let usage_before = memory_management.get_memory_usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management.reserve(size, &[]);
        }
        let usage_after = memory_management.get_memory_usage();
        // Check that we haven't increased our memory usage significantly
        assert!(usage_after.bytes_reserved <= (usage_before.bytes_reserved as f64 * 1.1) as usize);
    }

    // Test pools without slices. More or less same as tests above.
    #[test]
    fn noslice_test_handle_mutability() {
        let mem_props = MemoryDeviceProperties {
            max_page_size: 128 * 1024 * 1024,
            memory_alignment: 32,
        };
        let mut memory_management = DynamicMemoryManagement::from_configuration(
            BytesStorage::default(),
            mem_props,
            MemoryConfiguration::WithoutSubSlices,
        );
        let handle = memory_management.reserve(10, &[]);
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test]
    fn noslice_alloc_two_chunk() {
        let page_size = 2048;

        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::NoSlices,
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &[]);
        let _new_handle = memory_management.reserve(alloc_size, &[]);

        let usage = memory_management.get_memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test]
    fn noslice_alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size: 512,
                chunk_num_prealloc: 0,
                pool_type: PoolType::NoSlices,
            }],
            32,
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &[]);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size, &[]);

        let usage = memory_management.get_memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, alloc_size);
    }

    #[test]
    fn noslice_alloc_allocs_new_storage() {
        let page_size = 1024;
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::NoSlices,
            }],
            32,
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size, &[]);
        let _new_handle = memory_management.reserve(alloc_size, &[]);
        let usage = memory_management.get_memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test]
    fn noslice_alloc_respects_alignment_size() {
        let page_size = 500;
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size,
                chunk_num_prealloc: 0,
                pool_type: PoolType::NoSlices,
            }],
            50,
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size, &[]);
        let _new_handle = memory_management.reserve(alloc_size, &[]);
        let usage = memory_management.get_memory_usage();
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
                pool_type: PoolType::Slices(size),
            })
            .collect();
        let mut memory_management =
            DynamicMemoryManagement::new(BytesStorage::default(), pools, 10);
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s, &[]));
        let usage = memory_management.get_memory_usage();
        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<usize>());
    }

    #[test]
    fn noslice_worst_case_allocation() {
        let mut memory_management = DynamicMemoryManagement::new(
            BytesStorage::default(),
            vec![MemoryPoolOptions {
                page_size: 1000000,
                chunk_num_prealloc: 0,
                pool_type: PoolType::NoSlices,
            }],
            32,
        );
        // Allocate from small to big. This currently requires a new buffer each time.
        let sizes = [100, 1000, 10000, 100000, 1000000];
        let _handles: Vec<_> = sizes
            .iter()
            .map(|&size| memory_management.reserve(size, &[]))
            .collect();
        let usage = memory_management.get_memory_usage();
        assert_eq!(usage.bytes_in_use, sizes.iter().sum::<usize>());
        assert_eq!(usage.bytes_reserved, sizes.iter().sum::<usize>());
    }

    #[test]
    fn noslice_allocate_deallocate_reallocate() {
        let mut memory_management = DynamicMemoryManagement::from_configuration(
            BytesStorage::default(),
            MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                memory_alignment: 32,
            },
            MemoryConfiguration::WithoutSubSlices,
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &[]))
            .collect();
        let usage_before = memory_management.get_memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &[]))
            .collect();
        let usage_after = memory_management.get_memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }
}
