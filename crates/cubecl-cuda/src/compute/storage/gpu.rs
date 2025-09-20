use crate::compute::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{
    ComputeStorage, StorageHandle, StorageId, StorageUtilization, VirtualStorage,
};
use cudarc::driver::DriverError;
use cudarc::driver::sys::*;
use std::collections::{HashMap, VecDeque};

/// Buffer storage for NVIDIA GPUs.
///
/// This struct manages memory resources for CUDA kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    memory: HashMap<StorageId, cudarc::driver::sys::CUdeviceptr>,
    deallocations: Vec<StorageId>,
    stream: cudarc::driver::sys::CUstream,
    ptr_bindings: PtrBindings,
    mem_alignment: usize,
}

/// A GPU memory resource allocated for CUDA using [GpuStorage].
#[derive(Debug)]
pub struct GpuResource {
    /// The GPU memory pointer.
    pub ptr: u64,
    /// The CUDA binding pointer.
    pub binding: *mut std::ffi::c_void,
    /// The size of the resource.
    pub size: u64,
}

impl GpuResource {
    /// Creates a new [GpuResource].
    pub fn new(ptr: u64, binding: *mut std::ffi::c_void, size: u64) -> Self {
        Self { ptr, binding, size }
    }
}

impl GpuStorage {
    /// Creates a new [GpuStorage] instance for the specified CUDA stream.
    ///
    /// # Arguments
    ///
    /// * `mem_alignment` - The memory alignment requirement in bytes.
    /// * `stream` - The CUDA stream for asynchronous memory operations.
    pub fn new(mem_alignment: usize, stream: CUstream) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            stream,
            ptr_bindings: PtrBindings::new(),
            mem_alignment,
        }
    }

    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    pub fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(ptr) = self.memory.remove(&id) {
                unsafe {
                    cudarc::driver::result::free_async(ptr, self.stream).unwrap();
                }
            }
        }
    }
}

unsafe impl Send for GpuResource {}
unsafe impl Send for GpuStorage {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStorage")
            .field("stream", &self.stream)
            .finish()
    }
}

/// Manages active CUDA buffer bindings in a ring buffer.
///
/// This ensures that pointers remain valid during kernel execution, preventing use-after-free errors.
struct PtrBindings {
    slots: Vec<cudarc::driver::sys::CUdeviceptr>,
    cursor: usize,
}

impl PtrBindings {
    /// Creates a new [PtrBindings] instance with a fixed-size ring buffer.
    fn new() -> Self {
        Self {
            slots: uninit_vec(crate::device::CUDA_MAX_BINDINGS as usize),
            cursor: 0,
        }
    }

    /// Registers a new pointer in the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The CUDA device pointer to register.
    ///
    /// # Returns
    ///
    /// A reference to the registered pointer.
    fn register(&mut self, ptr: u64) -> &u64 {
        self.slots[self.cursor] = ptr;
        let ptr_ref = self.slots.get(self.cursor).unwrap();

        self.cursor += 1;

        // Reset the cursor when the ring buffer is full.
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        ptr_ref
    }
}

impl ComputeStorage for GpuStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = self
            .memory
            .get(&handle.id)
            .expect("Storage handle not found");

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        GpuResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            size,
        )
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        let ptr = unsafe { cudarc::driver::result::malloc_async(self.stream, size as usize) };
        let ptr = match ptr {
            Ok(ptr) => ptr,
            Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => {
                return Err(IoError::BufferTooBig(size as usize));
            }
            Err(other) => {
                return Err(IoError::Unknown(format!(
                    "CUDA allocation error: {}",
                    other
                )));
            }
        };
        self.memory.insert(id, ptr);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    fn flush(&mut self) {
        self.perform_deallocations();
    }
}

type GpuVirtualAddress = CUdeviceptr;

struct MappingInfo {
    virtual_addr: GpuVirtualAddress,
    size: usize,
    handle: Vec<CUmemGenericAllocationHandle>,
}

pub struct GpuVirtualStorage {
    device_id: i32,
    mem_alignment: usize,
    physical_block_size: usize,
    reservations: HashMap<GpuVirtualAddress, usize>,
    physical_handles: VecDeque<CUmemGenericAllocationHandle>,
    mappings: HashMap<StorageId, MappingInfo>,
    ptr_bindings: PtrBindings,
}

impl GpuVirtualStorage {
    pub fn new(device_id: i32, granularity: usize, block_size: usize) -> Self {
        Self {
            device_id,
            mem_alignment: granularity,
            physical_block_size: block_size.next_multiple_of(granularity),
            reservations: HashMap::new(),
            physical_handles: VecDeque::new(),
            mappings: HashMap::new(),
            ptr_bindings: PtrBindings::new(),
        }
    }

    fn set_access_permissions(
        &self,
        virtual_addr: CUdeviceptr,
        size: usize,
        device_id: i32,
    ) -> Result<(), IoError> {
        unsafe {
            let mut access_desc: CUmemAccessDesc = std::mem::zeroed();
            access_desc.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id = device_id;
            access_desc.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            let result = cuMemSetAccess(virtual_addr, size, &access_desc, 1);

            match result {
                CUresult::CUDA_SUCCESS => Ok(()),
                other => Err(IoError::Unknown(format!(
                    "CUDA set access failed: {:?}",
                    other
                ))),
            }
        }
    }
}

impl VirtualStorage for GpuVirtualStorage {
    type VirtualAddress = GpuVirtualAddress;
    type Resource = GpuResource;

    fn granularity(&self) -> usize {
        self.mem_alignment
    }

    fn physical_block_size(&self) -> usize {
        self.physical_block_size
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = self
            .mappings
            .get(&handle.id)
            .expect("Storage handle not found")
            .virtual_addr;

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        GpuResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            size,
        )
    }

    fn reserve(&mut self, size: usize) -> Result<Self::VirtualAddress, IoError> {
        let aligned_size = size.next_multiple_of(self.mem_alignment);

        unsafe {
            let mut virtual_addr: CUdeviceptr = 0;

            let result = cuMemAddressReserve(&mut virtual_addr, aligned_size, 0, 0, 0);

            match result {
                CUresult::CUDA_SUCCESS => {
                    self.reservations.insert(virtual_addr, aligned_size);
                    Ok(virtual_addr)
                }
                CUresult::CUDA_ERROR_OUT_OF_MEMORY => Err(IoError::BufferTooBig(aligned_size)),
                other => Err(IoError::Unknown(format!(
                    "CUDA reserve failed: {:?}",
                    other
                ))),
            }
        }
    }

    fn alloc_physical(&mut self) -> Result<(), IoError> {
        unsafe {
            let mut mem_handle: CUmemGenericAllocationHandle = 0;
            let mut prop: CUmemAllocationProp = std::mem::zeroed();

            prop.type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.requestedHandleTypes = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
            prop.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;
            prop.win32HandleMetaData = std::ptr::null_mut();
            prop.allocFlags.compressionType = 0;

            let result = cuMemCreate(&mut mem_handle, self.physical_block_size, &prop, 0);

            match result {
                CUresult::CUDA_SUCCESS => {
                    self.physical_handles.push_back(mem_handle);
                    Ok(())
                }
                CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
                    Err(IoError::BufferTooBig(self.physical_block_size))
                }
                other => Err(IoError::Unknown(format!(
                    "CUDA alloc physical failed: {:?}",
                    other
                ))),
            }
        }
    }

    fn map(
        &mut self,
        start_address: Self::VirtualAddress,
        size: usize,
    ) -> Result<StorageHandle, IoError> {
        let aligned_size = size.next_multiple_of(self.mem_alignment);
        let blocks_needed = aligned_size.div_ceil(self.physical_block_size);

        // Verificar que tenemos suficientes bloques físicos disponibles
        if self.physical_handles.len() < blocks_needed {
            return Err(IoError::Unknown(format!(
                "Insufficient physical memory blocks. Need: {}, Available: {}",
                blocks_needed,
                self.physical_handles.len()
            )));
        }

        let handles: Vec<_> = self.physical_handles.drain(..blocks_needed).collect();

        // Map each block
        let mut mapped_ranges = Vec::with_capacity(blocks_needed);
        let mut current_addr = start_address;

        for (i, handle) in handles.iter().enumerate() {
            unsafe {
                let result = cuMemMap(current_addr, self.physical_block_size, 0, *handle, 0);
                match result {
                    CUresult::CUDA_SUCCESS => {
                        if let Err(e) = self.set_access_permissions(
                            current_addr,
                            self.physical_block_size,
                            self.device_id,
                        ) {
                            // Rollback all successful mappings.
                            for &addr in &mapped_ranges {
                                cuMemUnmap(addr, self.physical_block_size);
                            }
                            // Return all handles.
                            for h in handles {
                                self.physical_handles.push_back(h);
                            }
                            return Err(e);
                        }
                        mapped_ranges.push(current_addr);
                        current_addr += self.physical_block_size as u64;
                    }
                    other => {
                        for &addr in &mapped_ranges {
                            cuMemUnmap(addr, size);
                        }

                        for h in handles {
                            self.physical_handles.push_back(h);
                        }
                        return Err(IoError::Unknown(format!("CUDA map failed: {:?}", other)));
                    }
                }
            }
        }

        let id = StorageId::new();
        self.mappings.insert(
            id,
            MappingInfo {
                virtual_addr: start_address,
                size: aligned_size,
                handle: handles,
            },
        );

        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset: 0,
                size: aligned_size as u64,
            },
        ))
    }

    fn unmap(&mut self, id: StorageId) {
        if let Some(mapping) = self.mappings.remove(&id) {
            let mut current_addr = mapping.virtual_addr;
            let num_blocks = mapping.size.div_ceil(self.physical_block_size);
            for block in 0..num_blocks {
                unsafe {
                    cuMemUnmap(current_addr, self.physical_block_size);
                }
                current_addr += self.physical_block_size as u64;
            }

            // Return all handles to the pool.
            for handle in mapping.handle {
                self.physical_handles.push_back(handle);
            }
        }
    }


    // Cleanup does not free reservations as they can be later reused.
    fn cleanup(&mut self) {
        unsafe {
            for (_, mapping) in self.mappings.drain() {
                cuMemUnmap(mapping.virtual_addr, mapping.size);
                for handle in mapping.handle {
                    cuMemRelease(handle);
                }
            }

            for handle in self.physical_handles.drain(..) {
                cuMemRelease(handle);
            }
        }
    }
}


impl Drop for GpuVirtualStorage {
    fn drop(&mut self) {
        self.cleanup();
        for (addr, size) in self.reservations.drain() {
               unsafe { cuMemAddressFree(addr, size)};
        }
    }
}
unsafe impl Send for GpuVirtualStorage {}

#[cfg(test)]
mod vmm_tests {
    use super::*;
    use cubecl_runtime::storage::{StorageHandle, StorageId, StorageUtilization, VirtualStorage};
    use std::collections::HashSet;

    fn create_test_storage() -> GpuVirtualStorage {
        cudarc::driver::result::init().unwrap();
        let device_id = 0;
        let device_ptr = cudarc::driver::result::device::get(device_id).unwrap();
        let granularity = query_min_granularity(device_id);

        GpuVirtualStorage::new(device_id, granularity, granularity)
    }

    fn query_min_granularity(device_id: i32) -> usize {
        unsafe {
            let mut prop: CUmemAllocationProp = std::mem::zeroed();
            prop.type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device_id;
            prop.requestedHandleTypes = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
            prop.win32HandleMetaData = std::ptr::null_mut();
            prop.allocFlags.compressionType = 0;

            let mut granularity: usize = 0;
            let res = cuMemGetAllocationGranularity(
                &mut granularity as *mut usize,
                &prop,
                CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            );
            if res != CUresult::CUDA_SUCCESS {
                panic!("cuMemGetAllocationGranularity failed: {:?}", res);
            }
            granularity
        }
    }

    // Helper function to get total device memory
    fn get_device_total_memory(device_id: i32) -> usize {
        unsafe {
            let mut total_mem: usize = 0;
            let mut free_mem: usize = 0;

            // Switch to the correct device context
            let device = cudarc::driver::result::device::get(device_id).unwrap();
            let ctx = cudarc::driver::result::primary_ctx::retain(device).unwrap();
             cudarc::driver::result::ctx::set_current(ctx).unwrap();

            let result = cudarc::driver::sys::cuMemGetInfo_v2(&mut free_mem, &mut total_mem);
            if result != CUresult::CUDA_SUCCESS {
                // Fallback: use a reasonable default if query fails
                println!("Warning: Could not query device memory, using 8GB default: {:?}", result);
                return 8 * 1024 * 1024 * 1024; // 8GB default
            }

            total_mem
        }
    }

    #[test]
    fn test_basic_reserve_allocate_map_workflow() {
        let mut storage = create_test_storage();

        // Test basic workflow: reserve -> alloc_physical -> map
        let size = 8192; // 8KB

        // Step 1: Reserve virtual address space
        let virtual_addr = storage.reserve(size).expect("Failed to reserve memory");
        assert_ne!(virtual_addr, 0, "Virtual address should not be null");

        // Verify reservation was recorded
        assert!(storage.reservations.contains_key(&virtual_addr));
        assert_eq!(
            storage.reservations[&virtual_addr],
            size.next_multiple_of(storage.granularity())
        );

        // Step 2: Allocate physical memory
        storage
            .alloc_physical()
            .expect("Failed to allocate physical memory");

        // Verify physical handle was created
        assert_eq!(
            storage.physical_handles.len(),
            1,
            "Should have one physical handle"
        );

        // Step 3: Map virtual to physical
        let handle = storage
            .map(virtual_addr, size)
            .expect("Failed to map memory");

        // Verify mapping was created
        assert!(storage.mappings.contains_key(&handle.id));
        assert_eq!(storage.mappings[&handle.id].virtual_addr, virtual_addr);
        assert_eq!(
            storage.mappings[&handle.id].size,
            size.next_multiple_of(storage.granularity())
        );

        // Verify physical handle was consumed
        assert_eq!(
            storage.physical_handles.len(),
            0,
            "Physical handle should be consumed"
        );

        // Step 4: Test getting resource
        let resource = storage.get(&handle);
        assert_eq!(resource.ptr, virtual_addr + handle.offset());
        assert_eq!(resource.size, handle.size());

        // Step 5: Test cleanup
        storage.unmap(handle.id);
        assert!(
            !storage.mappings.contains_key(&handle.id),
            "Mapping should be removed"
        );
    }

    #[test]
    fn test_physical_memory_reuse() {
        let mut storage = create_test_storage();
        let granularity = storage.granularity();

        // Reserve virtual address space
        let addr = storage
            .reserve(granularity * 1000)
            .expect("Failed to reserve first address");

        // Allocate multiple physical blocks
        storage
            .alloc_physical()
            .expect("Failed to allocate first block");
        storage
            .alloc_physical()
            .expect("Failed to allocate second block");
        storage
            .alloc_physical()
            .expect("Failed to allocate third block");

        assert_eq!(
            storage.physical_handles.len(),
            3,
            "Should have 3 physical handles"
        );
        let size = granularity;
        // Map first allocation
        let handle1 = storage
            .map(addr, size)
            .expect("Failed to map first allocation");

        let addr2 = addr + size as u64;

        assert_eq!(
            storage.physical_handles.len(),
            2,
            "Should have 2 handles remaining after first map"
        );
        assert_eq!(storage.mappings.len(), 1, "Should have 1 mapping");

        // Map second allocation (using 2x size to consume 2 blocks)
        let handle2 = storage
            .map(addr2, size * 2)
            .expect("Failed to map second allocation");

        assert_eq!(
            storage.physical_handles.len(),
            0,
            "Should have 0 handles remaining after second map"
        );
        assert_eq!(storage.mappings.len(), 2, "Should have 2 mappings");

        // Verify both mappings are active
        let resource1 = storage.get(&handle1);
        let resource2 = storage.get(&handle2);

        assert_eq!(resource1.ptr, addr + handle1.offset());
        assert_eq!(resource1.size, handle1.size());
        assert_eq!(resource2.ptr, addr2 + handle2.offset());
        assert_eq!(resource2.size, handle2.size());

        // Test memory reuse: unmap first allocation
        storage.unmap(handle1.id);

        assert_eq!(storage.mappings.len(), 1, "Should have 1 mapping remaining");
        assert_eq!(
            storage.physical_handles.len(),
            1,
            "Should have 1 handle returned to pool"
        );

        // Unmap second allocation (should return 2 handles since it used 2 blocks)
        storage.unmap(handle2.id);

        assert_eq!(storage.mappings.len(), 0, "Should have no mappings");
        assert_eq!(
            storage.physical_handles.len(),
            3,
            "Should have all 3 handles returned to pool"
        );

        // Test reuse: allocate new mapping to verify handles are reused
        let addr3 = addr + (size * 3) as u64;
        let handle3 = storage
            .map(addr3, size)
            .expect("Failed to map with reused handle");

        assert_eq!(
            storage.physical_handles.len(),
            2,
            "Should have 2 handles remaining after reuse"
        );
        assert_eq!(storage.mappings.len(), 1, "Should have 1 new mapping");

        // Verify the reused mapping works correctly
        let resource3 = storage.get(&handle3);
        assert_eq!(resource3.ptr, addr3 + handle3.offset());
        assert_eq!(resource3.size, handle3.size());

        // Test multi-block reuse: map with size that requires 2 blocks
        let addr4 = addr + (size * 4) as u64;
        let handle4 = storage
            .map(addr4, size * 2)
            .expect("Failed to map multi-block with reused handles");

        assert_eq!(
            storage.physical_handles.len(),
            0,
            "Should have no handles remaining after multi-block reuse"
        );
        assert_eq!(storage.mappings.len(), 2, "Should have 2 mappings");

        // Verify multi-block mapping works
        let resource4 = storage.get(&handle4);
        assert_eq!(resource4.ptr, addr4 + handle4.offset());
        assert_eq!(resource4.size, handle4.size());

        // Final cleanup verification
        storage.unmap(handle3.id);
        storage.unmap(handle4.id);

        assert_eq!(
            storage.mappings.len(),
            0,
            "Should have no mappings after final cleanup"
        );
        assert_eq!(
            storage.physical_handles.len(),
            3,
            "Should have all handles returned after final cleanup"
        );
    }

    #[test]
    fn test_virtual_memory_overcommit() {
        let mut storage = create_test_storage();
        let granularity = storage.granularity();

        // Get total device memory
        let device_total_memory = get_device_total_memory(storage.device_id);
        println!(
            "Device total memory: {} GB",
            device_total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let overcommit_size = device_total_memory * 2;
        let aligned_overcommit_size = overcommit_size.next_multiple_of(granularity);

        println!(
            "Attempting to reserve {} GB of virtual memory (2x device memory)",
            aligned_overcommit_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );


        let virtual_addr1 = storage
            .reserve(aligned_overcommit_size)
            .expect("Failed to reserve virtual memory exceeding device memory");

        assert_ne!(virtual_addr1, 0, "Virtual address should not be null");
        assert!(storage.reservations.contains_key(&virtual_addr1));
        assert_eq!(
            storage.reservations[&virtual_addr1],
            aligned_overcommit_size
        );
        // Verify we now have virtual memory totaling 5x device memory reserved
        let total_reserved: usize = storage.reservations.values().sum();
        let expected_total = aligned_overcommit_size;
        assert_eq!(total_reserved, expected_total);

        println!(
            "Successfully reserved {} GB total virtual memory ({}x device memory)",
            total_reserved as f64 / (1024.0 * 1024.0 * 1024.0),
            total_reserved as f64 / device_total_memory as f64
        );
    }
}
