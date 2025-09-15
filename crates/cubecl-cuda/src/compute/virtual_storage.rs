
/// Internal representation of a CUDA virtual memory range.
///
/// This structure tracks a contiguous virtual address space that has been reserved
/// from the CUDA driver. It maintains both the address range information and the
/// current mapping status to prevent invalid operations.
///
/// # State Management
///
/// The `is_mapped` field prevents double-mapping errors and ensures proper
/// resource lifecycle management. Virtual spaces must be unmapped before they
/// can be released.
struct CudaMemoryRange {
    /// Base virtual address of the reserved memory range
    start: CUdeviceptr,
    /// Size of the virtual address range in bytes
    size: u64,
    /// Whether this virtual space is currently mapped to physical memory
    is_mapped: bool,
}

impl CudaMemoryRange {
    /// Creates a new virtual memory range in the unmapped state.
    ///
    /// # Arguments
    /// * `start` - Base virtual address returned by CUDA's address reservation
    /// * `size` - Size of the reserved address range in bytes
    fn new(start: CUdeviceptr, size: u64) -> Self {
        Self {
            start,
            size,
            is_mapped: false,
        }
    }

    fn size(&self) -> u64 {
        self.size
    }

    /// Marks this virtual space as mapped to physical memory.
    ///
    /// This should be called after successful `cuMemMap` operations to maintain
    /// consistent state tracking.
    fn set_mapped(&mut self) {
        self.is_mapped = true;
    }

    /// Marks this virtual space as unmapped from physical memory.
    ///
    /// This should be called after successful `cuMemUnmap` operations to allow
    /// for future remapping or release of the virtual address space.
    fn set_unmapped(&mut self) {
        self.is_mapped = false;
    }
}

/// Internal representation of a CUDA physical memory allocation.
///
/// This structure wraps CUDA's generic allocation handle and tracks the mapping
/// status to prevent invalid operations such as double-mapping or freeing
/// mapped memory.
///
/// # Resource Safety
///
/// Physical handles maintain their own mapping state independently of virtual
/// spaces. This allows for validation that prevents resource leaks and ensures
/// proper cleanup ordering.
struct CudaPhysicalHandle {
    /// CUDA generic allocation handle for the physical memory
    handle: CUmemGenericAllocationHandle,
    /// Size of the physical allocation in bytes
    size: u64,
    /// Whether this physical memory is currently mapped to a virtual space
    is_mapped: bool,
}

impl CudaPhysicalHandle {
    /// Creates a new physical memory handle in the unmapped state.
    ///
    /// # Arguments
    /// * `handle` - CUDA allocation handle returned by `cuMemCreate`
    /// * `size` - Size of the physical allocation in bytes
    fn new(handle: CUmemGenericAllocationHandle, size: u64) -> Self {
        Self {
            handle,
            size,
            is_mapped: false,
        }
    }

    fn size(&self) -> u64 {
        self.size
    }

    /// Marks this physical memory as mapped to a virtual address space.
    ///
    /// This prevents the physical memory from being freed while it's in use
    /// and enables proper resource lifecycle validation.
    fn set_mapped(&mut self) {
        self.is_mapped = true;
    }

    /// Marks this physical memory as unmapped from virtual address spaces.
    ///
    /// This allows the physical memory to be freed or remapped to different
    /// virtual address spaces.
    fn set_unmapped(&mut self) {
        self.is_mapped = false;
    }
}

/// CUDA implementation of virtual memory storage.
///
/// This implementation provides a complete virtual memory management system using
/// CUDA's virtual memory APIs. It manages the full lifecycle of virtual address
/// spaces, physical memory allocations, and their mappings.
///
/// # Architecture
///
/// The storage maintains three primary data structures:
///
/// 1. **Reserved Spaces**: Virtual address ranges reserved from the driver
/// 2. **Physical Handles**: Hardware memory allocations from the GPU
/// 3. **Active Mappings**: Connections between virtual spaces and physical memory
///
/// # Resource Management
///
/// All CUDA resources are automatically cleaned up when the storage is dropped,
/// but manual cleanup via `cleanup()` is also supported for precise resource
/// control. The implementation maintains strict state tracking to prevent
/// resource leaks and invalid operations.
///
/// # Thread Safety
///
/// This implementation is marked as `Send` because CUDA handles and addresses
/// are thread-safe from the CUDA runtime's perspective. However, users must
/// ensure proper CUDA context management across threads.
pub struct CudaVirtualStorage {
    /// CUDA device ID where all memory operations are performed
    device_id: i32,
    /// Reserved virtual address spaces indexed by their unique identifiers
    /// These represent address ranges that have been reserved but may not be mapped
    reserved_spaces: HashMap<VirtualSpaceId, CudaMemoryRange>,
    /// Physical memory allocations indexed by their unique identifiers
    /// These represent actual GPU memory that may or may not be mapped
    physical_handles: HashMap<PhysicalStorageId, CudaPhysicalHandle>,
    /// Active mappings between virtual and physical memory
    /// These represent the connections that make memory accessible to kernels
    active_mappings: HashMap<StorageId, VirtualMapping>,
    /// Storage IDs that have been marked for deallocation
    /// These will be processed during the next `flush()` operation
    deallocations: Vec<StorageId>,
    /// Memory allocation granularity in bytes (typically the GPU page size)
    /// All allocations must be aligned to this boundary
    granularity: usize,
}

impl CudaVirtualStorage {
    /// Creates a new CUDA virtual storage instance for the specified device.
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID where memory operations will be performed
    /// * `granularity` - Allocation granularity in bytes (should match GPU page size)
    ///
    /// # Notes
    /// The granularity should be obtained by querying the device properties at runtime
    /// using `cuMemGetAllocationGranularity` to ensure optimal performance.
    fn new(device_id: i32, granularity: usize) -> Self {
        Self {
            device_id,
            reserved_spaces: HashMap::new(),
            physical_handles: HashMap::new(),
            active_mappings: HashMap::new(),
            deallocations: Vec::new(),
            granularity,
        }
    }

    /// Sets memory access permissions for a mapped virtual address range.
    ///
    /// This is a required step after mapping memory in CUDA's virtual memory system.
    /// Without proper permissions, memory access will fail at runtime.
    ///
    /// # Arguments
    /// * `addr` - Base virtual address of the mapped region
    /// * `size` - Size of the mapped region in bytes
    /// * `device_id` - CUDA device that should have access to this memory
    fn set_access_permissions(&self, addr: CUdeviceptr, size: u64, device_id: i32) {
        let access_desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };

        unsafe {
            cuMemSetAccess(addr, size as usize, &access_desc, 1);
        }
    }

    /// Creates a new CUDA physical memory allocation.
    ///
    /// This function directly interfaces with CUDA's virtual memory API to create
    /// a physical memory allocation that can be mapped to virtual address spaces.
    /// The allocation is platform-specific and uses appropriate handle types.
    ///
    /// # Arguments
    /// * `size` - Size in bytes of the physical memory to allocate
    ///
    /// # Returns
    /// * `Ok(CudaPhysicalHandle)` - Handle to the allocated physical memory
    /// * `Err(IoError::BufferTooBig)` - If allocation fails due to insufficient memory
    ///
    /// # Error Handling
    /// CUDA allocation failures are converted to `IoError::BufferTooBig` to maintain
    /// consistency with the storage trait interface.
    fn create_cuda_handle(&self, size: u64) -> Result<CudaPhysicalHandle, IoError> {
        let mut handle = 0;
        let handle_type = {
            #[cfg(unix)]
            {
                CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            }
            #[cfg(target_os = "windows")]
            {
                CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_WIN32
            }
        };

        let prop = CUmemAllocationProp {
            type_: CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: handle_type,
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device_id,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: Default::default(),
        };

        if unsafe { cuMemCreate(&mut handle, size as usize, &prop, 0).result() }.is_err() {
            return Err(IoError::BufferTooBig(size as usize));
        }

        Ok(CudaPhysicalHandle::new(handle, size))
    }

    /// Immediately releases all allocated resources.
    ///
    /// This function performs emergency cleanup of all CUDA resources managed by
    /// this storage instance. It's automatically called by the `Drop` implementation
    /// but can be called manually for precise resource management.
    ///
    /// # Cleanup Order
    /// Resources are cleaned up in dependency order to avoid CUDA errors:
    /// 1. Unmap all active virtual-to-physical mappings
    /// 2. Free all reserved virtual address spaces
    /// 3. Release all physical memory allocations
    ///
    /// # Error Handling
    /// CUDA API errors during cleanup are ignored to prevent panics during
    /// resource cleanup. This follows RAII principles where cleanup should
    /// not fail in destructors.
    fn cleanup(&mut self) {
        // Step 1: Unmap all active mappings
        for (_, mapping) in self.active_mappings.drain() {
            if let Some(virtual_space) = self.reserved_spaces.get(&mapping.virtual_id) {
                unsafe {
                    // Ignore errors during cleanup to prevent panics
                    let _ = cuMemUnmap(virtual_space.start, virtual_space.size as usize);
                }
            }
        }

        // Step 2: Free all reserved virtual address spaces
        for (_, space) in self.reserved_spaces.drain() {
            unsafe {
                // Ignore errors during cleanup to prevent panics
                let _ = cuMemAddressFree(space.start, space.size as usize);
            }
        }

        // Step 3: Release all physical handles
        for (_, handle) in self.physical_handles.drain() {
            unsafe {
                cuMemRelease(handle.handle);
            }
        }
    }
}

impl Drop for CudaVirtualStorage {
    /// Automatic resource cleanup when the storage goes out of scope.
    ///
    /// This ensures that all CUDA resources are properly released even if
    /// the user forgets to call `cleanup()` manually. Following RAII principles,
    /// this prevents resource leaks in normal program execution.
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl VirtualStorage for CudaVirtualStorage {
    type Resource = CudaResource;

    fn alignment(&self) -> usize {
        self.granularity
    }

    /// Reserves a contiguous virtual address range from the CUDA driver.
    ///
    /// This operation only reserves address space without allocating physical memory
    /// or establishing any mappings. The reserved space appears in the process's
    /// virtual address space but will cause access violations if used before mapping.
    ///
    /// # Implementation Details
    /// - Automatically aligns the requested size to the device granularity
    /// - Uses CUDA's `cuMemAddressReserve` with automatic address selection
    /// - Creates internal tracking structures for the reserved space
    /// - Returns a handle that can be used for future mapping operations
    ///
    /// # Error Conditions
    /// - Returns `IoError::BufferTooBig` if the virtual address space cannot be reserved
    /// - This typically indicates virtual address space exhaustion or driver limits
    fn try_reserve(&mut self, total_size: usize) -> Result<VirtualAddressSpaceHandle, IoError> {
        let virtual_size = total_size.next_multiple_of(self.granularity);

        let mut base_address = 0;
        if unsafe {
            cuMemAddressReserve(
                &mut base_address,
                virtual_size,
                0, // Preferred address (0 = automatic)
                0,
                0,
            )
            .result()
        }
        .is_err()
        {
            return Err(IoError::BufferTooBig(total_size));
        }

        let id = VirtualSpaceId::new();
        let virtual_range = CudaMemoryRange::new(base_address, virtual_size as u64);

        self.reserved_spaces.insert(id, virtual_range);

        Ok(VirtualAddressSpaceHandle::new(
            id,
            StorageUtilization {
                size: virtual_size as u64,
                offset: 0,
            },
        ))
    }

    /// Converts a storage handle into a usable resource for GPU operations.
    ///
    /// This function performs the critical translation from abstract storage handles
    /// to concrete memory resources that can be used by GPU kernels and operations.
    /// It looks up the active mapping and constructs a resource with the correct
    /// virtual memory address.
    ///
    /// # Resource Construction
    /// The returned `CudaResource` follows existing [`CudaStorage`] implementation to make it compatible
    /// with he current architecture. It contains:
    /// - The virtual memory pointer adjusted for any offset
    /// - A binding pointer for kernel parameter passing
    /// - Size and offset information for bounds checking
    ///
    /// # Error Handling
    /// This function panics on invalid handles rather than returning errors because
    /// invalid handles indicate programming bugs that should be caught during
    /// development rather than handled at runtime.
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let mapping = self.active_mappings.get(&handle.id)
            .expect("Storage handle not found in active mappings. Handle may be unmapped, deallocated, or invalid");

        let offset = handle.offset();
        let size = handle.size();

        let virtual_space = self.reserved_spaces.get(&mapping.virtual_id)
            .expect("The virtual address of this mapping is not found. Likely the reserved space for this mapping has already been released");

        let ptr = virtual_space.start + offset;

        CudaResource::new(ptr, ptr as *mut std::ffi::c_void, offset, size)
    }

    /// Allocates physical memory from the GPU without establishing any mappings.
    ///
    /// This operation directly interfaces with the CUDA driver to allocate actual
    /// GPU memory. The allocated memory exists in the GPU's physical address space
    /// but is not accessible until it's mapped to a virtual address space.
    ///
    /// # Allocation Strategy
    /// - Automatically aligns requested size to device granularity for optimal performance
    /// - Creates CUDA generic allocation handles that can be mapped multiple times
    /// - Uses device-specific allocation properties for optimal memory placement
    /// - Tracks allocation state to prevent invalid operations
    ///
    /// # Performance Considerations
    /// Physical allocations are expensive operations that may involve GPU driver
    /// communication. Consider batching allocations or using memory pools for
    /// frequently allocated sizes.
    fn alloc(&mut self, size: u64) -> Result<PhysicalStorageHandle, IoError> {
        let aligned_size = size.next_multiple_of(self.granularity as u64);

        let physical_handle = self.create_cuda_handle(aligned_size)?;
        let id = PhysicalStorageId::new();

        self.physical_handles.insert(id, physical_handle);

        Ok(PhysicalStorageHandle::new(
            id,
            StorageUtilization {
                offset: 0,
                size: aligned_size,
            },
        ))
    }

    /// Marks a storage mapping for deallocation during the next flush operation.
    ///
    /// This function implements lazy deallocation to improve performance by batching
    /// expensive GPU driver operations. The mapping remains valid until `flush()`
    /// is called, allowing for potential optimizations such as memory reuse.
    ///
    /// # Deallocation Strategy
    /// Rather than immediately unmapping and freeing resources, this function:
    /// - Adds the storage ID to a deallocation queue
    /// - Leaves all mappings and allocations intact
    /// - Defers actual cleanup until `flush()` is called
    ///
    /// # Usage Pattern
    /// This design allows memory pools to:
    /// - Batch multiple deallocations for efficiency
    /// - Potentially reuse recently deallocated memory
    /// - Control when expensive GPU operations occur
    fn dealloc(&mut self, id: StorageId) {
        if self.active_mappings.get_mut(&id).is_some() {
            self.deallocations.push(id);
        }
    }

    /// Processes all pending deallocations and releases resources to the GPU.
    ///
    /// This function performs the actual cleanup work for all storage IDs that
    /// have been marked for deallocation. It completely removes mappings, frees
    /// virtual address spaces, and releases physical memory back to the GPU.
    ///
    /// # Cleanup Process
    /// For each pending deallocation:
    /// 1. Remove the virtual-to-physical mapping
    /// 2. Unmap and free the virtual address space
    /// 3. Release the physical memory allocation
    /// 4. Update internal tracking structures
    ///
    /// # Error Handling
    /// CUDA API errors during flush are ignored to ensure that partial cleanup
    /// can still occur. This prevents one failed operation from blocking the
    /// cleanup of other resources.
    fn flush(&mut self) {
        for id in &self.deallocations {
            if let Some(mapping) = self.active_mappings.remove(id) {
                // Free virtual address space
                if let Some(virtual_space) = self.reserved_spaces.remove(&mapping.virtual_id) {
                    unsafe {
                        cuMemUnmap(virtual_space.start, virtual_space.size as usize);
                        cuMemAddressFree(virtual_space.start, virtual_space.size as usize);
                    }
                }
                for physical_id in &mapping.physical_ids {
                    // Release physical memory
                    if let Some(physical) = self.physical_handles.remove(physical_id) {
                        unsafe {
                            cuMemRelease(physical.handle);
                        }
                    }
                }
            }
        }
        self.deallocations.clear();
    }

    /// Establishes a mapping between virtual address space and physical memory.
    ///
    /// This is the core operation that makes memory accessible to GPU programs.
    /// It connects a reserved virtual address space with an allocated physical
    /// memory block, enabling actual memory access through virtual addresses.
    ///
    /// # Mapping Process
    /// 1. Validates that both virtual space and physical memory exist and are unmapped
    /// 2. Performs the CUDA mapping operation using `cuMemMap`
    /// 3. Sets appropriate memory access permissions for the device
    /// 4. Updates internal state tracking for both virtual and physical resources
    /// 5. Creates a new storage handle for accessing the mapped memory
    ///
    /// # State Validation
    /// The function includes assertions to prevent invalid operations:
    /// - Virtual address spaces cannot be double-mapped
    /// - Physical memory cannot be mapped multiple times simultaneously
    /// - These checks prevent resource corruption and aid in debugging
    ///
    /// # Error Recovery
    /// If the CUDA mapping operation fails, all state changes are reverted
    /// and an error is returned, ensuring consistent internal state.
    fn try_map(
        &mut self,
        virtual_addr: VirtualAddressSpaceHandle,
        handles: Vec<PhysicalStorageHandle>,
    ) -> Result<StorageHandle, IoError> {
        let virtual_id = virtual_addr.id();

        // Get mutable references to both the virtual space and physical handle
        let virtual_space = self
            .reserved_spaces
            .get(&virtual_id)
            .ok_or(IoError::InvalidHandle)?;

        // Validate that neither resource is already mapped
        assert!(
            !virtual_space.is_mapped,
            "Try map function received an invalid virtual space: This address space is already mapped. Cannot remap without previous unmapping"
        );

        let mut address = virtual_space.start;
        let mut size = 0;
        let mut physical_ids = Vec::with_capacity(handles.len());
        for handle in &handles {
            let physical_id = handle.id();

            let cuda_handle = self
                .physical_handles
                .get_mut(&physical_id)
                .ok_or(IoError::InvalidHandle)?;

            size += cuda_handle.size();

            assert!(
                !cuda_handle.is_mapped,
                "Try map function received an invalid physical memory handle. This physical handle is already mapped. Cannot remap without previous unmapping"
            );

            assert!(
                size <= virtual_space.size(),
                "Try map function received invalid parameters: Handle size cannot be bigger than total reserved address space, but total handle size is {} and virtual space size is {}.",
                size,
                virtual_space.size()
            );

            // Perform the CUDA mapping operation
            unsafe {
                let result = cuMemMap(
                    address,
                    cuda_handle.size() as usize,
                    0, // offset in the handle
                    cuda_handle.handle,
                    0, // flags
                );
                if result != CUDA_SUCCESS {
                    let mut rollback_address = virtual_space.start;
                    for prev_id in physical_ids.iter() {
                        let prev = self
                            .physical_handles
                            .get(prev_id)
                            .ok_or(IoError::InvalidHandle)?;

                        if let Err(e) = cuMemUnmap(rollback_address, prev.size() as usize).result()
                        {
                            panic!("Rollback failed in try_map operation: {}.", e)
                        }

                        rollback_address += prev.size();
                    }
                    return Err(IoError::InvalidHandle);
                }
            }

            // Update state tracking
            cuda_handle.set_mapped();

            // Set memory access permissions for the device
            self.set_access_permissions(address, handle.size(), self.device_id);
            address += handle.size();
            physical_ids.push(physical_id);
        }

        let virtual_space_mut = self
            .reserved_spaces
            .get_mut(&virtual_id)
            .ok_or(IoError::InvalidHandle)?;
        virtual_space_mut.set_mapped();

        // Create storage handle for the new mapping
        let storage_id = StorageId::new();
        let mapping = VirtualMapping::new(physical_ids, virtual_id);
        self.active_mappings.insert(storage_id, mapping);

        Ok(StorageHandle::new(
            storage_id,
            StorageUtilization {
                offset: 0,
                size: virtual_space_mut.size(),
            },
        ))
    }

    /// Removes the mapping between virtual address space and physical memory.
    ///
    /// This operation breaks the connection between virtual and physical memory
    /// without releasing either resource. After unmapping, the virtual address
    /// space becomes inaccessible but remains reserved, and the physical memory
    /// remains allocated but unmapped.
    ///
    /// # Unmapping Process
    /// 1. Locates the mapping using the provided storage ID
    /// 2. Retrieves both virtual space and physical handle references
    /// 3. Performs the CUDA unmap operation using `cuMemUnmap` on each handle
    /// 4. Updates state tracking to mark resources as unmapped
    /// 5. Removes the mapping from active tracking
    ///
    /// # Error Recovery
    /// If the CUDA unmap operation fails, the mapping is re-inserted to maintain
    /// consistent state, and an error is returned to the caller.
    /// All unmapped handles are remapped to their old addresses (rollbacking) for consistency.
    fn try_unmap(&mut self, id: StorageId) -> Result<(), IoError> {
        if let Some(mapping) = self.active_mappings.remove(&id) {
            let virtual_space = self
                .reserved_spaces
                .get_mut(&mapping.virtual_id)
                .ok_or(IoError::InvalidHandle)?;

            let mut address = virtual_space.start;

            // In memory rollback journal in case of failure
            let mut unmapped_track: Vec<PhysicalStorageId> =
                Vec::with_capacity(mapping.physical_ids.len());

            for handle_id in mapping.physical_ids.iter() {
                let handle = self
                    .physical_handles
                    .get_mut(handle_id)
                    .ok_or(IoError::InvalidHandle)?;
                unsafe {
                    let result = cuMemUnmap(address, handle.size() as usize);
                    if result != CUDA_SUCCESS {
                        // rollback mapped handles in case of failure
                        let mut rollback_addr = virtual_space.start;
                        for prev_id in unmapped_track.iter() {
                            let prev = self
                                .physical_handles
                                .get_mut(prev_id)
                                .ok_or(IoError::InvalidHandle)?;

                            if let Err(e) =
                                cuMemMap(rollback_addr, prev.size() as usize, 0, prev.handle, 0)
                                    .result()
                            {
                                panic!("Rollback cuMemMap failed: {}", e);
                            }

                            rollback_addr += prev.size();
                        }

                        // Re-insert mapping into active_mappings
                        self.active_mappings.insert(id, mapping);
                        return Err(IoError::InvalidHandle);
                    }
                    unmapped_track.push(*handle_id);
                }

                // Update state
                address += handle.size();
                handle.set_unmapped();
            }
        }
        Ok(())
    }

    /// Releases a virtual address space back to the CUDA driver.
    ///
    /// This operation frees the virtual address range, making it available for
    /// future allocations by this or other processes. The virtual address space
    /// must be unmapped before it can be released.
    ///
    /// # Preconditions
    /// The virtual address space must not be currently mapped to physical memory.
    /// Attempting to release a mapped virtual space will result in an error and
    /// the space will remain allocated.
    ///
    /// # Error Handling
    /// - Returns `IoError::InvalidHandle` if the virtual space is still mapped
    /// - Returns `IoError::InvalidHandle` if the CUDA free operation fails
    /// - On error, the virtual space is re-inserted to maintain consistent state
    ///
    /// # Resource Management
    /// After successful release, the virtual space ID becomes invalid and should
    /// not be used for future operations.
    fn try_release(&mut self, id: VirtualSpaceId) -> Result<(), IoError> {
        if let Some(space) = self.reserved_spaces.remove(&id) {
            if space.is_mapped {
                // Cannot release a mapped virtual space
                self.reserved_spaces.insert(id, space);
                return Err(IoError::InvalidHandle);
            }

            unsafe {
                let res = cuMemAddressFree(space.start, space.size as usize);
                if res != CUDA_SUCCESS {
                    // Re-insert the space if free failed
                    self.reserved_spaces.insert(id, space);
                    return Err(IoError::InvalidHandle);
                }
            }
        }

        Ok(())
    }

    /// Frees physical memory back to the GPU.
    ///
    /// This operation permanently releases physical memory back to the CUDA driver,
    /// making it available for future allocations. The physical memory must be
    /// unmapped from all virtual address spaces before it can be freed.
    ///
    /// # Preconditions
    /// The physical memory must not be currently mapped to any virtual address space.
    /// Attempting to free mapped physical memory will result in an error and the
    /// memory will remain allocated.
    ///
    /// # Memory Recovery
    /// After successful freeing, the physical memory is returned to the GPU's
    /// memory pool and can be allocated by future operations. The physical
    /// storage ID becomes invalid and should not be used.
    ///
    /// # Error Handling
    /// - Returns `IoError::InvalidHandle` if the physical memory is still mapped
    /// - Returns `IoError::InvalidHandle` if the CUDA release operation fails
    /// - On error, the physical handle is re-inserted to maintain consistent state
    fn try_free(&mut self, id: PhysicalStorageId) -> Result<(), IoError> {
        if let Some(handle) = self.physical_handles.remove(&id) {
            if handle.is_mapped {
                // Cannot free mapped physical memory
                self.physical_handles.insert(id, handle);
                return Err(IoError::InvalidHandle);
            }

            unsafe {
                let res = cuMemRelease(handle.handle);
                if res != CUDA_SUCCESS {
                    // Re-insert the handle if release failed
                    self.physical_handles.insert(id, handle);
                    return Err(IoError::InvalidHandle);
                }
            }
        }
        Ok(())
    }
}

// Safety: CudaVirtualStorage can be safely sent between threads as long as
// CUDA context is properly managed. The handles and addresses are thread-safe
// from CUDA's perspective, but users must ensure proper CUDA context setup.
unsafe impl Send for CudaVirtualStorage {}
