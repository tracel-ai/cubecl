// crates/cubecl-cuda/src/compute/vmm.rs
// RM: Added VMM support for CUBECL-CUDA-
use cubecl_core::server::IoError;
use cubecl_runtime::storage::StorageId;
use cudarc::driver::sys::{
    CUdevice, CUdeviceptr, CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    CUmemAccessDesc, CUmemAllocationGranularity_flags, CUmemAllocationHandleType,
    CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, CUmemAllocationProp,
    CUmemAllocationType, CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
    CUmemGenericAllocationHandle, CUmemLocation, CUmemLocationType,
    CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE, cuDeviceGet, cuMemAddressFree,
    cuMemAddressReserve, cuMemCreate, cuMemGetAllocationGranularity, cuMemGetInfo_v2, cuMemMap,
    cuMemRelease, cuMemSetAccess, cuMemUnmap, cudaError_enum::CUDA_SUCCESS,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// GPU page size (2MB) used by CUDA VMM
///
/// CUDA VMM works in terms of GPU pages. All allocations and mappings
/// must be aligned to this size. This is a hardware constraint.
pub const GPU_PAGE_SIZE: u64 = 2 * 1024 * 1024;

macro_rules! cuda_driver_check {
    ($expr:expr) => {{
        let result = unsafe { $expr };
        if result != CUDA_SUCCESS {
            panic!("CUDA driver error at {}:{}: {:?}", file!(), line!(), result);
        }
    }};
}

macro_rules! cuda_driver_try {
    ($expr:expr, $err:expr) => {{
        let result = unsafe { $expr };
        if result != CUDA_SUCCESS {
            return Err($err);
        }
    }};
}

fn get_min_granularity(device: CUdevice) -> usize {
    let mut granularity: usize = 0;
    let prop = CUmemAllocationProp {
        type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        location: CUmemLocation {
            type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: Default::default(),
    };

    cuda_driver_check!(cuMemGetAllocationGranularity(
        &mut granularity,
        &prop,
        CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    ));

    granularity
}

// Utility to check the device current memory usage.
// Can panic if the CudaContext is not initiated.
// Returns a tuple with the free memory and the total available memory in bytes.
// NOT SURE IF THIS SHOULD BE A METHOD OF CUDA SERVER. PLEASE REVIEW THIS PART
fn get_device_memory_info() -> (u64, u64) {
    // Consultar memoria
    let mut free_bytes = 0usize;
    let mut total_bytes = 0usize;

    cuda_driver_check!(cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes));
    (free_bytes as u64, total_bytes as u64)
}

/// VMM allocation strategy configuration
///
/// This determines whether to use regular allocation strategy (provided by cudastorage)
// or try to use vmm with expandable segments
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Use regular CudaStorage.
    /// This should be preferred mostly during training. On Neural Networks, parameters and activations are typically fixed in size during training and therefore the expandable segments might be an overkill on this situation. However, having benchmarks to test both strategies would be nice (Might work on that later).
    Disabled,

    ExpandableSegments {
        /// Size of each physical memory segment.
        handle_size: u64, // The intuition behind this parameters is the following:
        // Handles are always aligned up to the GPU page size. Typically, this correspondes to about 2MiB. Therefore, it is not a good idea to configure the allocation strategy with a handle size that is smaller than that.
        virtual_size: u64, // Total virtual address space to reserve. The total number of handles that you will be able to create will be virtual_size / handle_size.
    },
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::Disabled
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HandleRange {
    start_idx: usize,
    length: usize,
    is_mapped: bool,
}


impl HandleRange {

    fn new(start_idx:usize, length: usize) -> Self {
        Self {
            start_idx,
            length,
            is_mapped: false
        }
    }

}

struct HandleManager {
    handles: Vec<Option<CUmemGenericAllocationHandle>>, // Pool of handles.
    // To guarantee that handles are contiguous in memory, we allocate them in ranges and map them to virtual addresses based on the index on the vector of handles. Therefore the handle associated VA is computed: addr = index * handle_size. This has the disadvantage that the handle size must be fixed (That is, physical memory allocations are fixed size), but it is not a problem since we can expand the size of an expandable block ([`resize`]) by adding new handles to it.
    free_ranges: BTreeMap<usize, BTreeSet<HandleRange>>, // length -> ranges
    allocated_ranges: HashMap<usize, HandleRange>,       // start_id of the range -> range
    handle_size: usize,
}

impl HandleManager {
    fn new(handle_size: usize, max_handles: usize) -> Self {
        Self {
            handles: vec![None; max_handles], // Preallocate the vector since we know the max number of handles from the beginning
            free_ranges: BTreeMap::new(),
            allocated_ranges: HashMap::new(),
            handle_size,
        }
    }

    /// Create a physical memory handle
    ///
    /// This allocates actual GPU memory that can be mapped to virtual addresses.
    /// Uses POSIX file descriptors as the handle type. In PyTorch impl, they try to use FABRIC handle types, which are more optimized for sharing across devices (RDMA).
    // However, I still have to take a look at burn-communication to see if there are implications there, so for now I decided to keep it this way for this first test.
    fn create_handle() -> Result<CUmemGenericAllocationHandle, IoError> {
        let mut handle = 0;

        let prop = CUmemAllocationProp {
            type_: CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device_id,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: Default::default(),
        };

        cuda_driver_try!(
            cuMemCreate(&mut handle, self.handle_size as usize, &prop, 0),
            IoError::BufferTooBig(self.handle_size as usize)
        );

        Ok(handle)
    }

    fn remove_handle(&mut self, idx: usize) -> Option<CUmemGenericAllocationHandle> {
        if let Some(handle) = self.handles[idx].take() {
            cuda_driver_check!(cuMemRelease(handle));
        }
    }

    fn insert_handle(&mut self, idx: usize) -> Result<(), IoError> {
        if let Some(handle) = self.handles[idx].take() {
            // There is already a handle at that position, re-insert and return an error.
            // If we do not do that two handles might have assigned the same address
            self.handles[idx] = Some(handle);
            return Err(IoError::InvalidHandle);
        }

        let new_handle = self.create_new_handle()?;
        self.handles[idx] = Some(new_handle);
        Ok(())
    }



    // Unmaps a range of handles.
    fn unmap_range(&mut self, start_idx: usize, base_addr: CUdeviceptr) -> Result<(), IoError> {
        if let Some(mut range) = self.allocated_ranges.get_mut(&start_idx) {
            // Fail early if already unmapped.
            if self.state != BlockState::Unmapped {
                return Err(IoError::InvalidHandle);
            }

            for (i, idx) in (range.start_idx..(range.start_idx + range.length)).enumerate() {
                // Use get_handle to get a reference to the handle
                match self.get_handle(idx) {
                    Ok(handle) => {
                        // Handles of the same range are contiguous in memory
                        let virtual_addr = base_addr + (i as u64 * handle_size);
                        // Note: I am not sure if it is a good idea to let return an error here in case of failure on the device side. It is what PyTorch does, so I assume there should not be any issues (See the macro C10_CUDA_DRIVER_CHECK  at https://github.dev/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
                        cuda_driver_try!(
                            cuMemUnmap(virtual_addr, handle_size as usize),
                            IoError::InvalidHandle
                        );
                        range.is_mapped = false;
                    }
                    Err(_) => {
                        // Handle is None.
                        // Here we have to return an error since handles must be allocated at this point.
                        return Err(IoError::InvalidHandle);
                    }
                }
            }
        }
    }

    // Maps a range of handles to their virtual addresses.
    // Doing it here ensures:
    // A) Handles are always contiguous in memory.
    // B) No allocated handles are None in the handles vec.
    fn map_range(&mut self, start_idx: usize, base_addr: CUdeviceptr) -> Result<(), IoError> {
        if let Some(mut range) = self.allocated_ranges.get_mut(&start_idx) {
            // Fail early if already mapped.
            if range.is_mapped {
                return Err(IoError::InvalidHandle);
            }

            // Iterate over the found range.
            for (i, idx) in (range.start_idx..(range.start_idx + range.length).enumerate()) {
                // Use get_handle to get a reference to the handle
                match self.get_handle(idx) {
                    Ok(handle) => {
                        // Handles of the same range are contiguous in memory
                        let virtual_addr = base_addr + (i as u64 * self.handle_size);
                        cuda_driver_try!(
                            cuMemMap(virtual_addr, self.handle_size as usize, handle),
                            IoError::InvalidHandle
                        );
                        range.is_mapped = true; // Set mapped
                    }
                    Err(_) => {
                        // Handle is None.
                        // Here we have to return an error since handles must be allocated at this point.
                        return Err(IoError::InvalidHandle);
                    }
                }
            }

            Ok(())
        } else {
            // The range does not exist
            Err(IoError::InvalidHandle)
        }
    }

    fn get_handle(&self, idx: usize) -> Result<&CUmemGenericAllocationHandle, IoError> {
        let handle = self.handles.get(idx).ok_or(IoError::InvalidHandle);
        Ok(handle)
    }

    fn clear(&mut self) {
        for mut handle in self.handles.drain(..) {
            if let Some(hand) = handle.take() {

                cuda_driver_check!(cuMemRelease(hand.handle));
            }
        }

        // Clear the handles vector and reset the HandleManager state
        self.free_ranges.clear();
        self.allocated_ranges.clear();
    }

    // Find the first free range of at least `required_count` handles
    // Returns None if there are no handles available.
    fn find_free_range(&self, required_count: usize) -> Option<HandleRange> {
        for (&len, ranges) in self.free_ranges.range(required_count..) {
            if let Some(range) = ranges.iter().next() {
                return Some(*range);
            }
        } // No free ranges available. The caller should return an IOError here
        None
    }

    // Alloc a range of a specified size, reusing handles if possible.
    // Returns the start idx of the first handle.
    // Note that no real allocation happens here. The real stuff happens at the insert_handle function. This is critical for lazy allocation. The allocation will happen when the physical memory is actually required by the expandable block.
    fn alloc_range(&mut self, count: usize) -> Result<usize, IOError> {
        let range = self
            .find_free_range(count)
            .ok_or(Err(IoError::InvalidHandle));

        // Remove range from free_ranges
        let ranges = self.free_ranges.get_mut(&range.length).unwrap();
        ranges.remove(&range);

        // If there are no more free ranges of that specified length, remove the key.
        if ranges.is_empty() {
            self.free_ranges.remove(&range.length);
        }

        let allocated = HandleRange::new(
            range.start_idx,
            count,
        );
        self.allocated_ranges.insert(range.start_idx, allocated);

        // If there are leftover ranges in the freelist, re-insert them
        if range.length > count {
            let leftover = HandleRange::new(
               range.start_idx + count,
               range.length - count,
            );
            self.free_ranges
                .entry(leftover.length)
                .or_insert_with(BTreeSet::new)
                .insert(leftover);
        }
        Ok((allocated.start_idx))
    }



    // Removes the range from the hashmap of allocated ranges and puts it on the btreeset of free ranges (This way we can reuse ranges without having to make a lot of calls to cuMemReserve)
    fn free_range(&mut self, start_idx: usize) -> Result<(), String> {
        if let Some(range) = self.allocated_ranges.remove(&start_idx) {
            self.free_ranges
                .entry(range.length)
                .or_insert_with(BTreeSet::new)
                .insert(range);
            Ok(())
        } else {
            Err("Range not found".into())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BlockState {
    Unmapped,  // Virtual space reserved, no physical memory
    Mapped,    // Physical memory mapped, available for allocation
    Allocated, // In active use
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MemoryPressureLevel {
    Normal,
    Moderate,
    High,
    Critical,
}

#[derive(Debug)]
pub struct ExpandableBlock {
    base_addr: CUdeviceptr,
    size: u64,
    gc_count_base: u64,    // Garbage-Collection count when this block was inserted
    allocation_count: u64, // How many times has this block been reused.
    state: BlockState, // Unmapped -> Mapped -> Allocated.
    handle_range: Option<usize>, // This is a pointer to a mapped range allocated from the
    // HandleManager. This way if we wanted to expand a block, we just de-map this range and try to map again a bigger one, which has almost zero-cost as it does not move physical memory at all.

}

impl Default for ExpandableBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpandableBlock {
    pub fn new() -> Self {

        Self {
            base_addr: 0,
            size: 0,
            state: BlockState::Unmapped,
            gc_count_base: 0,
            allocation_count: 0,
            range: None
        }
    }


    // Does not allocate actual memory, just creates a new block with an assigned memory range of the specified size.
    fn init(start_idx: usize, size: u64, handle_size: usize, manager: &mut HandleManager) -> Result<Self, IoError> {
        let base_addr = start_idx * handle_size;
        let required_handles = size / handle_size;

        let range_start = manager.alloc_range(required_handles)?;
        Self {
            base_addr,
            size,
            state: BlockState::Unmapped,
            gc_count_base: 0,
            allocation_count: 0,
            range: Some(range_start)
        }

    }

    //  Allocate physical memory for this block.
    fn alloc(&self, manager: &mut Handleanager) -> Result<(), IOError> {

    }

    // Deallocate physical memory for this block.
    fn dealloc(&self, manager: &mut Handleanager) -> Result<(), IOError> {

    }

    fn is_initalized(&self) -> bool {
        self.range.is_some()
    }

    fn map(&mut self, manager: &mut HandleManager) -> Result<(), IOError> {
        if !self.is_initalized || self.state != BlockState::Unmapped {
            return Err(IOError::InvalidHandle);
        }

        manager.map_range(self.range.unwrap(), // Wont panic here as we already checked if is some
        self.base_addr

    )?;
        self.state = BlockState::Mapped;
        Ok(())

    }

    fn unmap(&mut self, manager: &mut HandleManager) -> Result<(), IOError> {
        if self.state == BlockState::Allocated {
            return Err(IOError::InvalidHandle);
        }

        if !self.is_initalized || self.state == BlockState::Unmapped {
            return Ok(()); // Do not fail here
        }

        manager.unmap_range(self.range.unwrap(), // Wont panic here as we already checked if is some
        self.base_addr

    )?;
        self.state = BlockState::Unmapped;
        Ok(())

    }


    // Expands the current block to a new target size by appending a new range of handles at the end.
    // Note that the handles might not be fully contiguous in memory here, which might be less efficient but still valid. As long as we make sure that handle size is a multiple of the granularity, memory fragmentation should be minimal.
    fn expand(&mut self, new_size: u64, handle_size: usize, manager: &mut HandleManager) -> Result<(), IOError>{
        match self.state {

            BlockState::Mapped => {
            self.unmap(manager)?;
            },
            BlockState::Allocated => {

                return Err(IOError::InvalidHandle)
            },
            _ => {}

        };

        if let Some(range) = self.range {
            manager.free_range(range)?;
            self.range = None // Deallocate current range
        };

        let required_handles = new_size / handle_size;
        let start_idx = manager.alloc_range(required_handles)?;
        self.range = Some(start_idx);
        self.size = new_size;
        self.map(manager);
        Ok(())

    }


    fn f


    pub fn device_ptr(&self) -> CUdeviceptr {
        self.base_addr
    }

    /// Compute the age of the block for garbage collection
    pub fn gc_age(&self, current_gc_count: u64) -> u64 {
        current_gc_count.saturating_sub(self.gc_count_base)
    }

    /// Marks the block as used and updates its garbage collection counter.
    pub fn mark_used(&mut self, current_gc_count: u64) {
        self.allocation_count += 1;
        self.gc_count_base = current_gc_count;
    }



}

/// [`Expandable Storage`]
///
/// Inspired by PyTorch's expandable segments. The idea is to reserve a large
/// virtual address (VA) range once and *expand* it on demand by mapping
/// additional physical pages into that range. This avoids repeatedly
/// allocating slightly-larger chunks as shapes fluctuate (e.g. batched
/// inference), which would otherwise create many small, unrecoverable
/// fragments ("slivers").
///
/// # Regular (non-expandable) strategy
///
/// For large allocations (e.g. > 2 MiB), the regular allocator requests device memory
/// sized exactly to the user request (e.g. via `cudaMalloc`/HIP equivalent).
/// Freed blocks are cached and reused when a future request fits exactly or
/// in even multiples. This works well during training, especially with fixed batch size and static model
/// architecture, as tensor shapes (weights, activations, gradients, optimizer buffers)
/// generally stay the same across iterations.
///
/// # Motivation for this new implementation
/// I refer to the source code of CUDACachingAllocator if you want to verify my explanation of this problem:
/// github.dev/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp
///
/// Batched inference often changes batch size between iterations. Consider:
///
/// - Iteration 1: batch size **N**
/// - Iteration 2: batch size **N - 1**
/// - Iteration 3: batch size **N + 1**
///
/// On iteration 1 we allocate buffers sized for **N** (e.g. `N*A`, `N*A*B`, …).
/// If iteration 2 later uses **N - 1**, most of those `N`-sized blocks remain
/// "big enough" and can be reused. But iteration 3 (**N + 1**) forces the
/// allocator to request *slightly larger* blocks (`(N+1)*A`, `(N+1)*A*B`, …).
///
/// Because layers differ in size, some `(N+1)*A` requests may squeeze into
/// previously cached `N*A*B` segments (they're "big enough" but not exact).
/// As the model runs layer by layer, we partially fill many of these old
/// segments and leave small unusable tails at the end of each:
///
/// ```text
/// [ N*A*B segment ]: [ used by (N+1)*A ][  free sliver  ]
///                                           ^ too small for the next request
/// ```
///
/// With deep models, this repeats and we can accumulate dozens of
/// slivers across segments. When a new `(N+1)*A*B` block is needed and total
/// *free* memory exists only as scattered tails, a fresh large allocation is
/// required. If the device is near capacity, this can fail even though the
/// sum of free bytes would be sufficient *if* they were contiguous.
///
/// # Expandable segments and the VMM api:
///
/// Expandable segments allows the allocator to create one slarge virtual address (VA) range up front, and then
/// filling it (commiting) with actual device memory only as needed.
///
/// Since v. 10.2, NVIDIA provides primitives to develop these growing chunks of mapped memory:
// https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
///
/// 1. [`cuMemCreate`] creates a handle for a big range of physical device memory.
/// 2. [`cuMemAddressReserve`] reserves a virtual address range from CUDA.
/// 3. [`cuMemMap`] maps the physical handle to the virtual address retrieved by CUDA.
///
/// The freeing counterparts for these functions are:
///
/// 1. [`cuMemUnMap`] unmaps an entire VA range.
/// 2. [`cuMemRelease`] invalidates the handle, returning it to the OS.
/// 3. [`cuMemAddressFree`] returns the VA to CUDA.
///
/// The allocation workflow would be:
///
/// -> CuMemAddressReserve -> Ask CUDA for a new VA.
/// -> If Returned OK -> cuMemCreate to create a physical handle and cuMemMap to map it to the reservation.
/// Use [`CuMemSetAccess`] to define the access permissions for the mapped memory chunk.
///
/// The deallocation workflow is:
///
/// -> CuMemUnmap -> Unmaps the handle to the Virtual Address. As it does not destroy physical memory, we can map it again later for reuse (Important to set again the permissions via [`CuMemSetAccess`])
/// The idea then is to keep a pool of freed physical handles, which can then be remapped to VAs
pub struct ExpandableStorage {
    device_id: i32,
    base_address: CUdeviceptr,
    virtual_size: u64,
    handle_size: u64,
    handle_manager: HandleManager,
    blocks: HashMap<StorageId, ExpandableBlock>,

    // Free pools
    unmapped_blocks: Vec<StorageId>,
    mapped_blocks: Vec<StorageId>,

    gc_count: u64,               // Global garbage collection counter
    total_allocated_memory: u64, // Total allocated memory for this segment.
    gc_threshold_ratio: f64,     // Threshold to trigger garbage collection.
                                 // Any value between 0.6 and 0.7 should be okay
}

impl ExpandableStorage {
    /// This reserves virtual address space but doesn't allocate any physical memory yet.
    /// Physical memory will be allocated and mapped on-demand in alloc_and_map().
    pub fn new(
        device_id: i32,
        virtual_size: u64,
        handle_size: u64,
        gc_threshold_ratio: f64,
    ) -> Result<Self, IoError> {
        let mut base_address: CUdeviceptr = 0;

        let mut device: CUdevice = 0;
        cuda_driver_check!(cuDeviceGet(&mut device as *mut CUdevice, device_id));

        let granularity: u64 = match get_min_granularity(device).try_into() {
            // This can fail on x32 architectures, so we fallback to the hardcoded value (typically GPU page size is 2MiB)
            Ok(val) => val,
            Err(_) => GPU_PAGE_SIZE,
        };

        let handle_size = handle_size.next_multiple_of(granularity); // We need to ensure that the handles are multiples of the page size of the GPU for efficient physical allocations (https://forums.developer.nvidia.com/t/virtual-memory-management-minimum-granularity/268699)

        cuda_driver_try!(
            cuMemAddressReserve(&mut base_address, virtual_size as usize, 0, 0, 0),
            IoError::BufferTooBig(virtual_size as usize)
        );

        let max_handles = (virtual_size / handle_size) as usize;
        let mut handle_manager = HandleManager::new(handle_size, max_handles);

        Ok(Self {
            device_id,
            base_address,
            virtual_size,
            handle_size,
            handle_manager,
            blocks: HashMap::new(),
            unmapped_blocks: Vec::new(),
            mapped_blocks: Vec::new(),
            gc_count: 0,
            total_allocated_memory: 0,
            gc_threshold_ratio,
        })
    }

    // Obtain an allocated mapping by the block id on the blocks HashMap.
    // Refuses to return the block if it is not allocated.
    pub fn get_ptr(&self, storage_id: StorageId) -> Option<CUdeviceptr> {
        let block = self.blocks.get(&storage_id)?;

        if block.state != BlockState::Allocated {
            return None;
        }

        Some(block.device_ptr())
    }



    fn create_new_block(
        &mut self,
        aligned_size: u64,
        storage_id: StorageId,
    ) -> Result<(), IoError> {
        let required_handles = (aligned_size / self.handle_size) as usize;
        // Allocate handle indices using HandleManager
        let handle_indices = self
            .handle_manager
            .allocate_range(handle_count)
            .map_err(|_| IoError::BufferTooBig(aligned_size as usize))?;

        let mut new_handles = Vec::with_capacity(handle_count);

        for &index in handle_indices.iter() {
            // Create a new physical handle on GPU memory
            let handle = self.create_physical_handle(self.handle_size)?;
            new_handles.push(handle);
            self.handle_manager.handles[index] = Some(handle);
        }

        let virtual_addr = self.base_address + (handle_indices[0] as u64 * self.handle_size);

        // Perform the actual mapping. If it fails, will return and error and will trigger the retrial on the [`alloc_and_map`] method.
        self.map_handle_set(virtual_addr, &new_handles)?;

        let mut block = ExpandableBlock::new(virtual_addr, aligned_size, handle_indices);
        block.state = BlockState::Allocated;
        self.total_allocated_memory += block.size;
        block.mark_used(self.gc_count);
        self.blocks.insert(storage_id, block);

        Ok(())
    }

    // Utility methods to find blocks by size:
    fn find_mapped_block(&self, required_size: u64) -> Option<StorageId> {
        self.mapped_blocks
            .iter()
            .find(|&&id| {
                if let Some(block) = self.blocks.get(&id) {
                    block.size >= required_size && block.state == BlockState::Mapped
                } else {
                    false
                }
            })
            .copied()
    }

    fn find_unmapped_block(&self, required_size: u64) -> Option<StorageId> {
        self.unmapped_blocks
            .iter()
            .find(|&&id| {
                if let Some(block) = self.blocks.get(&id) {
                    block.size >= required_size && block.state == BlockState::Unmapped
                } else {
                    false
                }
            })
            .copied()
    }

    /// Function to reuse a mapped block.
    /// It is important to be able to reuse mapped blocks too because
    fn reuse_mapped_block(
        &mut self,
        reuse_id: StorageId,
        storage_id: StorageId,
    ) -> Result<CUdeviceptr, IoError> {
        // Remove old block
        let mut block = self
            .blocks
            .remove(&reuse_id)
            .ok_or(IoError::InvalidHandle)?;

        // Update block for new allocation
        block.state = BlockState::Allocated;
        block.mark_used(self.gc_count);

        // Store under new storage_id
        self.blocks.insert(storage_id, block);
        self.mapped_blocks.retain(|&id| id != reuse_id);
        self.total_allocated_memory += mapping.size;

        Ok(())
    }

    pub fn alloc_and_map(&mut self, size: u64, storage_id: StorageId) -> Result<(), IoError> {
        let aligned_size = size.next_multiple_of(self.handle_size);

        // 1. Attempt to reuse mapped blocks.
        if let Some(reuse_id) = self.find_mapped_block(aligned_size) {
            return self.reuse_mapped_block(reuse_id, storage_id);
        }

        // 2. Attempt to remap unmapped blocks.
        if let Some(block_idx) = self.find_unmapped_block(aligned_size) {
            return self.remap_block(block_idx, storage_id, aligned_size);
        }

        // 3. Attempt to create a new block.
        match self.create_new_block(aligned_size, storage_id) {
            Ok(mapping) => Ok(mapping),
            Err(_) => {
                // If it fails, handle the failure elegantly and re-try
                if self.handle_allocation_failure(aligned_size)? {
                    // Here we skip the lookup for mapped blocks beacuse we already checked it.
                    if let Some(block_idx) = self.find_unmapped_block(aligned_size) {
                        return self.remap_block(block_idx, storage_id, aligned_size);
                    }
                    self.create_new_block(aligned_size, storage_id)
                } else {
                    Err(IoError::BufferTooBig(aligned_size as usize))
                }
            }
        }
        Ok(())
    }

    // Remaps an existing block to a new storage identifier.
    fn remap_block(
        &mut self,
        remap_id: StorageId,
        storage_id: StorageId,
        required_size: u64,
    ) -> Result<(), IoError> {
        // Remove the unmapped block from the HashMap
        let mut block = self
            .blocks
            .remove(&remap_id)
            .ok_or(IoError::InvalidHandle)?;

        // Verify the block is unmapped and has sufficient size
        if block.state != BlockState::Unmapped {
            // Put block back and return error
            self.blocks.insert(remap_id, block);
            return Err(IoError::InvalidHandle);
        }

        if block.size < required_size {
            self.blocks.insert(remap_id, block);
            return Err(IoError::BufferTooBig(required_size as usize));
        }

        // Collect the handles from the block.
        // The filter closure ensures the handle indices are valid (they exist on the handles vec).
        let handles_to_map: Vec<VmemHandle> = block
            .handle_indices
            .iter()
            .filter_map(|&idx| {
                self.handle_manager
                    .handles
                    .get(idx)
                    .and_then(|h| h.as_ref())
                    .copied()
            })
            .collect();

        // Perform the actual memory mapping on the GPU.
        // Will fail in case there is no memory available to perform the mapping.
        self.map_handle_set(block.virtual_addr, &handles_to_map)?;

        // Update tracking statistics and insert under the new mapped id.
        block.state = BlockState::Allocated;
        self.total_allocated_memory += block.size;
        block.mark_used(self.gc_count);

        self.unmapped_blocks.retain(|&idx| idx != remap_id);

        self.blocks.insert(storage_id, block);

        Ok(())
    }

    // Utility to perform an actual mapping between a virtual address and a set of handles.
    // Receives an iterator of <VmemHandle> and calls cudarc to perfomr the mapping on GPU memory.
    // In case of failure during any of the maps, rolls back all previously mapped handles to prevent memory leaks.
    // Also sets the access permissions for the mapped handle.
    fn map_handle_set(
        &self,
        virtual_addr: CUdeviceptr,
        handles: &[VmemHandle],
    ) -> Result<(), IoError> {
        let total_handles = handles.len();
        for (i, handle) in handles.iter().enumerate() {
            let addr = virtual_addr + (i as u64 * self.handle_size);
            let result = unsafe { cuMemMap(addr, self.handle_size as usize, 0, handle.handle, 0) };
            // Rollback and return and error
            if result != CUDA_SUCCESS {
                for j in 0..i {
                    let cleanup_addr = virtual_addr + (j as u64 * self.handle_size);
                    unsafe { cuMemUnmap(cleanup_addr, self.handle_size as usize) };
                }
                return Err(IoError::BufferTooBig(self.handle_size as usize));
            }
        }

        self.set_access_permissions(virtual_addr, total_handles as u64 * self.handle_size);
        Ok(())
    }

    // Releases a set of handles. Will panic in case of failure.
    fn release_handle_set(&self, handles: Vec<VmemHandle>) {
        for handle in handles.into_iter() {
            if handle.handle != 0 {
                // Check if the handle is valid
                cuda_driver_check!(cuMemRelease(handle.handle));
            }
        }
    }



    /// Sets memory access permissions for the mapped range
    ///
    /// After mapping, we need to explicitly grant read/write access to the device.
    /// This is separate from the mapping step in CUDA VMM.
    ///
    /// As commented above, I am not implementing sharing accross devices still.
    /// Therefore, my implementation only gives read/write access to the current device.
    /// In the future, I will add peer access to other devices too for data sharing over RDMA.
    fn set_access_permissions(&self, virtual_addr: CUdeviceptr, size: u64) {
        let access_desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device_id,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };

        cuda_driver_check!(cuMemSetAccess(virtual_addr, size as usize, &access_desc, 1));
    }

    fn handle_allocation_failure(&mut self, required_size: u64) -> Result<bool, IoError> {
        self.gc_count += 1; // Increment global allocation counter.
        let pressure = self.memory_pressure();

        match pressure {
            MemoryPressureLevel::Normal => Ok(false),
            MemoryPressureLevel::Moderate => self.garbage_collect(true),
            MemoryPressureLevel::High => self.release_blocks(required_size, true), // The flags here force to release actual physical resources to make up new space.
            MemoryPressureLevel::Critical => self.release_all(true),
        }
    }

    fn memory_pressure(&self) -> MemoryPressureLevel {
        let (free_memory, total_memory) = get_device_memory_info();
        let device_utilization = (total_memory - free_memory) as f64 / total_memory as f64;

        // Determine if we should garbage collect.
        match device_utilization {
            d if d < self.gc_threshold_ratio => MemoryPressureLevel::Normal,
            d if d < 0.80 => MemoryPressureLevel::Moderate,
            d if d < 0.95 => MemoryPressureLevel::High,
            _ => MemoryPressureLevel::Critical,
        }
    }

    fn should_gc(&self, id: StorageId, age_threshold: u64) -> bool {
        if let Some(block) = self.blocks.get(&id) {
            block.gc_age(self.gc_count) >= age_threshold
        } else {
            false
        }
    }

    fn garbage_collect(&mut self, force_free: bool) -> Result<bool, IoError> {
        if self.mapped_blocks.is_empty() {
            return Ok(false);
        }

        // Compute the average age of all blocks.
        let total_age: u64 = self
            .mapped_blocks
            .iter()
            .map(|&idx| self.blocks[&idx].gc_age(self.gc_count))
            .sum();

        let avg_age = total_age / self.mapped_blocks.len() as u64;
        let age_threshold = avg_age.max(2); // Minimum 2 generations to avoid freeing too recent blocks.

        // Collect blocks to unmap
        let blocks_to_unmap: Vec<StorageId> = self
            .mapped_blocks
            .iter()
            .filter(|id| self.should_gc(**id, age_threshold))
            .copied()
            .collect();

        let mut freed_any = false;
        for storage_id in blocks_to_unmap {
            if force_free {
                if self.dealloc_flush(storage_id).is_ok() {
                    freed_any = true;
                }
            } else if self.dealloc_unmap(storage_id).is_ok() {
                freed_any = true;
            }
        }

        Ok(freed_any)
    }

    // This strategy is more aggressive than the garbage collection one,
    // releases as much blocks as we need to perform the new resize.
    fn release_blocks(&mut self, required_size: u64, force_free: bool) -> Result<bool, IoError> {
        if self.mapped_blocks.is_empty() {
            return Ok(false);
        }

        // Sort blocks by age (oldest first)
        let mut aged_blocks: Vec<(StorageId, u64, u64)> = self
            .mapped_blocks
            .iter()
            .filter_map(|&id| {
                self.blocks
                    .get(&id)
                    .map(|block| (id, block.size, block.gc_age(self.gc_count)))
            })
            .collect();

        aged_blocks.sort_by(|a, b| b.2.cmp(&a.2)); // Sort by age descending

        let mut total_freed = 0u64;
        let mut freed_any = false;

        for (block_idx, block_size, _) in aged_blocks {
            if total_freed >= required_size {
                break;
            }
            if force_free {
                if self.dealloc_flush(block_idx).is_ok() {
                    total_freed += block_size;
                    freed_any = true;
                }
            } else if self.dealloc_unmap(block_idx).is_ok() {
                total_freed += block_size;
                freed_any = true;
            }
        }

        Ok(freed_any)
    }

    // Releases all blocks. The flag force_free actually releases the memory since cumemunmap will not actually clean up resources. Therefore, in cases of memory pressure, we want to dealloc_flush the blocks to make space for new physical memory.
    pub fn release_all(&mut self, force_free: bool) -> Result<bool, IoError> {
        let initial_count = self.mapped_blocks.len();
        if initial_count == 0 {
            return Ok(false);
        }

        // Get all mapped blocks for complete deallocation
        let blocks_to_release: Vec<StorageId> = self.mapped_blocks.clone();
        let mut freed_any = false;

        for storage_id in blocks_to_release {
            if force_free {
                if self.dealloc_flush(storage_id).is_ok() {
                    freed_any = true
                }
            } else if self.dealloc_unmap(storage_id).is_ok() {
                freed_any = true;
            }
        }

        Ok(freed_any)
    }

    // Deallocates an allocated block, transitioning to mapped state.
    // When it is called from the storage module, the block will transition from allocated -> mapped.
    // Complete deallocation will happen typically on garbage collection when memory pressure is high or via the flush method.
    // DESIGN DECISION IS TO KEEP ALL DEALLOCS AS SAFE-NO-OPS in case the storage id is not found.
    pub fn dealloc(&mut self, storage_id: StorageId) -> Result<(), IoError> {
        if let Some(block) = self.blocks.get_mut(&storage_id) {
            if block.state != BlockState::Allocated {
                return Ok(());
            }

            // Transition to mapped for reuse
            block.state = BlockState::Mapped;
            block.gc_count_base = self.gc_count;

            // Move to mapped pool
            self.mapped_blocks.push(storage_id);

            self.total_allocated_memory = self.total_allocated_memory.saturating_sub(block.size);
        };
        Ok(())
    }

    // Second deallocation step: unmap a block (Transition from mapped to unmapped state)
    pub fn dealloc_unmap(&mut self, storage_id: StorageId) -> Result<(), IoError> {
        {
            if let Some(block) = self.blocks.get(&storage_id) {
                if block.state == BlockState::Allocated {
                    self.dealloc(storage_id)?; // Transition to mapped.
                } else if block.state != BlockState::Mapped {
                    return Ok(());
                }
            } else {
                return Ok(()); // Safe NO-OP in case the block is not found
            }
        }

        let block = self
            .blocks
            .get_mut(&storage_id)
            .ok_or(IoError::InvalidHandle)?;

        // Again, i am not sure about failing here. Will add tests to check if failure is common
        cuda_driver_check!(cuMemUnmap(block.virtual_addr, block.size as usize));

        // Unmap the handles but keep physical memory
        block.unmap_handles(self.handle_size);

        // Update block state
        block.state = BlockState::Unmapped;

        // Move from mapped to unmapped pool
        self.mapped_blocks.retain(|&id| id != storage_id);
        self.unmapped_blocks.push(storage_id);

        Ok(())
    }

    pub fn clear_handles(&mut self) {
        self.handle_manager.clear();
    }

    /// Completely deallocates a block, releasing all handles, therefore returning memory to the device.
    pub fn dealloc_flush(&mut self, storage_id: StorageId) -> Result<(), IoError> {
        // First, dealloc and unmap the block
        {
            if let Some(block) = self.blocks.get(&storage_id) {
                // take a non-mutable reference as we just need to check the block 's state.

                if (block.state == BlockState::Mapped) || (block.state == BlockState::Allocated) {
                    self.dealloc_unmap(storage_id)?;
                }
            } else {
                return Ok(());
            }
        }

        // Now properly remove the block:
        let block = self
            .blocks
            .remove(&storage_id)
            .ok_or(IoError::InvalidHandle)?;

        // Get block handles.
        let mut handles_to_release = Vec::with_capacity(block.handle_indices.len());
        for &handle_idx in &block.handle_indices {
            if let Some(vmm_handle) = self.handle_manager.remove_handle(handle_idx) {
                handles_to_release.push(vmm_handle);
            }
        }

        // Free the handle range in HandleManager
        if let Some(&first_idx) = block.handle_indices.first() {
            self.handle_manager.free_range(first_idx);
        }

        // See ExpandableSegments::unmap_handles at https://github.dev/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp
        self.release_handle_set(handles_to_release);

        // Remove from all tracking pools
        self.mapped_blocks.retain(|&id| id != storage_id);
        self.unmapped_blocks.retain(|&id| id != storage_id);

        Ok(())
    }
}

impl Drop for ExpandableSegment {
    /// Clean up all resources when the segment is dropped
    ///
    /// This is critical for avoiding memory leaks. VMM requires explicit cleanup
    /// of mappings, physical handles, and virtual address space.
    ///
    /// Order matters here:
    /// 1. Unmap virtual->physical mappings
    /// 2. Release physical memory handles
    /// 3. Free virtual address space
    fn drop(&mut self) {
        // 1. Unmap all mapped blocks

        for key in self.blocks.keys().cloned().collect::<Vec<_>>() {
            self.dealloc_flush(key).unwrap_or_else(|e| {
                panic!("Failed to dealloc block with storage id {:?}: {}", key, e)
            });
        }

        self.clear_handles();

        // 3. Free the virtual address space reservation
        cuda_driver_check!(cuMemAddressFree(
            self.base_address,
            self.virtual_size as usize
        ));
    }
}

#[cfg(test)]
mod expandable_segment_tests {
    use super::*;
    use cudarc::driver::sys::cuInit;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn setup() {
        INIT.call_once(|| {
            // Initialize CUDA if needed for tests
            unsafe {
                cuInit(0);
            }
        });
    }

    // Test constants
    const TEST_DEVICE_ID: i32 = 0;
    const TEST_VIRTUAL_SIZE: u64 = 1 << 30; // 1 GB
    const TEST_HANDLE_SIZE: u64 = 2 << 20; // 2 MB
    const TEST_GC_THRESHOLD: f64 = 0.7;

    fn create_test_segment() -> ExpandableSegment {
        setup();
        ExpandableSegment::new(
            TEST_DEVICE_ID,
            TEST_VIRTUAL_SIZE,
            TEST_HANDLE_SIZE,
            TEST_GC_THRESHOLD,
        )
        .expect("Failed to create test segment")
    }

    #[test]
    fn test_segment_creation() {
        let segment = create_test_segment();

        assert_eq!(segment.device_id, TEST_DEVICE_ID);
        assert_eq!(segment.virtual_size, TEST_VIRTUAL_SIZE);
        assert_eq!(
            segment.handle_size,
            TEST_HANDLE_SIZE.next_multiple_of(GPU_PAGE_SIZE)
        );
        assert_eq!(segment.gc_threshold_ratio, TEST_GC_THRESHOLD);
        assert!(segment.base_address != 0, "Base address should be non-zero");
        assert_eq!(segment.total_allocated_memory, 0);
        assert_eq!(segment.gc_count, 0);
        assert!(segment.blocks.is_empty());
        assert!(segment.unmapped_blocks.is_empty());
        assert!(segment.mapped_blocks.is_empty());
    }

    #[test]
    fn test_unmap_and_remap() {
        let mut segment = create_test_segment();
        let storage_id1 = StorageId::new();
        let storage_id2 = StorageId::new();
        let alloc_size = 1024 * 1024;

        // Allocate and then unmap
        let mapping1 = segment.alloc_and_map(alloc_size, storage_id1).unwrap();
        let virtual_addr = mapping1.virtual_addr;

        segment.dealloc(storage_id1).unwrap();
        segment.dealloc_unmap(storage_id1).unwrap();

        // Verify unmapped state
        let block = segment.blocks.get(&storage_id1).unwrap();
        assert_eq!(block.state, BlockState::Unmapped);
        assert!(segment.unmapped_blocks.contains(&storage_id1));
        assert!(!segment.mapped_blocks.contains(&storage_id1));

        // Remap with new storage ID
        let mapping2 = segment.alloc_and_map(alloc_size, storage_id2).unwrap();

        assert_eq!(
            mapping2.virtual_addr, virtual_addr,
            "Should reuse same virtual address"
        );
        assert!(
            !segment.blocks.contains_key(&storage_id1),
            "Old storage ID should be removed"
        );
        assert!(
            segment.blocks.contains_key(&storage_id2),
            "New storage ID should exist"
        );

        let block = segment.blocks.get(&storage_id2).unwrap();
        assert_eq!(block.state, BlockState::Allocated);
    }

    #[test]
    fn test_size_alignment() {
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let unaligned_size = 1234567; // Not aligned to handle size

        let mapping = segment.alloc_and_map(unaligned_size, storage_id).unwrap();

        let expected_size = unaligned_size.next_multiple_of(segment.handle_size);
        assert_eq!(
            mapping.size, expected_size,
            "Size should be aligned to handle size"
        );

        let block = segment.blocks.get(&storage_id).unwrap();
        assert_eq!(block.size, expected_size);
    }

    #[test]
    fn test_multiple_handles_per_block() {
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let large_size = segment.handle_size * 3; // Requires 3 handles

        let _mapping = segment.alloc_and_map(large_size, storage_id).unwrap();

        let block = segment.blocks.get(&storage_id).unwrap();
        assert_eq!(block.handle_indices.len(), 3, "Should have 3 handles");

        // Verify handles are contiguous
        for i in 1..block.handle_indices.len() {
            assert_eq!(
                block.handle_indices[i],
                block.handle_indices[i - 1] + 1,
                "Handle indices should be contiguous"
            );
        }

        // Verify virtual address alignment
        let expected_addr =
            segment.base_address + (block.handle_indices[0] as u64 * segment.handle_size);
        assert_eq!(block.virtual_addr, expected_addr);
    }

    #[test]
    fn test_get_mapping() {
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let alloc_size = 1024 * 1024;

        // Before allocation
        assert!(
            segment.get_mapping(storage_id).is_none(),
            "Should return None for non-existent ID"
        );

        // After allocation
        let original_mapping = segment.alloc_and_map(alloc_size, storage_id).unwrap();
        let retrieved_mapping = segment.get_mapping(storage_id).unwrap();

        assert_eq!(
            retrieved_mapping.virtual_addr,
            original_mapping.virtual_addr
        );
        assert_eq!(retrieved_mapping.size, original_mapping.size);

        // After deallocation (goes to mapped state)
        segment.dealloc(storage_id).unwrap();
        assert!(
            segment.get_mapping(storage_id).is_none(),
            "Should return None for mapped blocks"
        );
    }

    #[test]
    fn test_garbage_collection() {
        let mut segment = create_test_segment();
        let storage_ids: Vec<StorageId> = (0..3).map(|_| StorageId::new()).collect();
        let alloc_size = 1024 * 1024;

        // Allocate multiple blocks
        for &storage_id in &storage_ids {
            segment.alloc_and_map(alloc_size, storage_id).unwrap();
        }

        // Deallocate all (they go to mapped state)
        for &storage_id in &storage_ids {
            segment.dealloc(storage_id).unwrap();
        }

        assert_eq!(segment.mapped_blocks.len(), 3);

        // Advance GC count to make blocks eligible for collection
        segment.gc_count += 10;

        // Run garbage collection
        let result = segment.garbage_collect(false);
        assert!(result.is_ok());

        // Should have unmapped some blocks
        assert!(
            segment.unmapped_blocks.len() > 0,
            "Some blocks should be unmapped"
        );
        assert!(
            segment.mapped_blocks.len() < 3,
            "Some blocks should be moved from mapped"
        );
    }

    #[test]
    fn test_gc_age_calculation() {
        // Validates the correct calculation of the gc_age field inside each block.
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let alloc_size = 1024 * 1024;

        // Allocate at gc_count = 0
        segment.alloc_and_map(alloc_size, storage_id).unwrap();
        segment.dealloc(storage_id).unwrap();

        let block = segment.blocks.get(&storage_id).unwrap();
        assert_eq!(block.gc_age(0), 0);
        assert_eq!(block.gc_age(5), 5);
        assert_eq!(block.gc_age(100), 100);
    }

    #[test]
    fn test_mark_used() {
        // Basically validates that when you call mark_used it updates the gc_age value of the block.
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let alloc_size = 1024 * 1024;

        segment.alloc_and_map(alloc_size, storage_id).unwrap();

        let mut block = segment.blocks.remove(&storage_id).unwrap();
        assert_eq!(block.allocation_count, 1);
        assert_eq!(block.gc_count_base, 0);

        segment.gc_count = 10;
        block.mark_used(segment.gc_count);

        assert_eq!(block.allocation_count, 2);
        assert_eq!(block.gc_count_base, 10);
        assert_eq!(block.gc_age(15), 5);

        segment.blocks.insert(storage_id, block);
    }

    #[test]
    fn test_release_blocks_by_size() {
        // Test for the second level garbage collection method
        let mut segment = create_test_segment();
        let storage_ids: Vec<StorageId> = (0..5).map(|_| StorageId::new()).collect();
        let alloc_size = 1024 * 1024;

        for &storage_id in storage_ids.iter() {
            segment.alloc_and_map(alloc_size, storage_id).unwrap();
        }

        assert_eq!(segment.mapped_blocks.len(), 0); // Should be empty as all blocks are allocated here.

        for (i, &storage_id) in storage_ids.iter().enumerate() {
            segment.dealloc(storage_id).unwrap();

            // Make some blocks older
            if i < 2 {
                let block = segment.blocks.get_mut(&storage_id).unwrap();
                block.gc_count_base = 0; // Make it older
            }
        }

        segment.gc_count = 10;
        let required_size = alloc_size * 3; // Need to free 2 blocks worth

        assert_eq!(
            segment.mapped_blocks.len(),
            5,
            "Should have 5 mapped blocks"
        );

        let result = segment.release_blocks(required_size, false);
        assert!(result.is_ok());

        // Should have freed enough to meet the requirement
        assert!(
            segment.unmapped_blocks.len() >= 2,
            "Should unmap at least 2 blocks"
        );
    }

    #[test]
    fn test_dealloc_flush() {
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let alloc_size = 1024 * 1024;

        // Allocate
        let _mapping = segment.alloc_and_map(alloc_size, storage_id).unwrap();
        let handle_count = segment
            .handle_manager
            .handles
            .iter()
            .filter(|h| h.is_some())
            .count();

        // Flush should completely remove the block and handles
        let result = segment.dealloc_flush(storage_id);
        assert!(result.is_ok(), "Flush should succeed");

        assert!(
            !segment.blocks.contains_key(&storage_id),
            "Block should be removed"
        );
        assert!(!segment.mapped_blocks.contains(&storage_id));
        assert!(!segment.unmapped_blocks.contains(&storage_id));

        // Handles should be cleaned up (though some may remain as None)
        let remaining_handles = segment
            .handle_manager
            .handles
            .iter()
            .filter(|h| h.is_some())
            .count();
        assert!(
            remaining_handles < handle_count,
            "Some handles should be released"
        );
    }

    #[test]
    fn test_double_dealloc_handling() {
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let alloc_size = 1024 * 1024;

        segment.alloc_and_map(alloc_size, storage_id).unwrap();

        // First dealloc should succeed
        assert!(segment.dealloc(storage_id).is_ok());

        // Second dealloc should be safe (no-op)
        let result = segment.dealloc(storage_id);
        assert!(result.is_ok(), "Double dealloc should be safe");
    }

    #[test]
    fn test_invalid_storage_id() {
        // Error handling test
        let mut segment = create_test_segment();
        let invalid_id = StorageId::new();

        // Operations on non-existent storage ID should return errors
        assert!(segment.get_mapping(invalid_id).is_none());
        assert!(segment.dealloc_unmap(invalid_id).is_ok());
        assert!(segment.dealloc_flush(invalid_id).is_ok());
    }

    // Validates storage level transitions are well rounded for blocks.
    #[test]
    fn test_state_transitions() {
        let mut segment = create_test_segment();
        let storage_id = StorageId::new();
        let alloc_size = 1024 * 1024;

        // Allocated -> Mapped -> Unmapped -> Back to Allocated
        let _mapping = segment.alloc_and_map(alloc_size, storage_id).unwrap();
        assert_eq!(segment.blocks[&storage_id].state, BlockState::Allocated);

        segment.dealloc(storage_id).unwrap();
        assert_eq!(segment.blocks[&storage_id].state, BlockState::Mapped);

        segment.dealloc_unmap(storage_id).unwrap();
        assert_eq!(segment.blocks[&storage_id].state, BlockState::Unmapped);

        // Should not be able to get mapping in unmapped state
        assert!(segment.get_mapping(storage_id).is_none());
    }
}
