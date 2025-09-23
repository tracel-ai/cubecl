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
    ptr_bindings: PtrBindings,
    stream: cudarc::driver::sys::CUstream,
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
    pub fn new(mem_alignment: usize, stream: cudarc::driver::sys::CUstream) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(),
            stream,
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
        f.debug_struct("GpuStorage").finish()
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

struct GpuVirtualAddressSpace {
    start_address: CUdeviceptr,
    size: u64,
    handles: Option<Vec<CUmemGenericAllocationHandle>>,
}

impl GpuVirtualAddressSpace {
    fn is_mapped(&self) -> bool {
        self.handles.is_some()
    }

    fn ptr(&self) -> CUdeviceptr {
        self.start_address
    }

    fn size(&self) -> u64 {
        self.size
    }
}

pub struct GpuVirtualStorage {
    device_id: i32,
    mem_alignment: usize,
    physical_block_size: u64,
    /// Reserved virtual memory ranges. Can be either mapped or unmapped.
    reservations: HashMap<StorageId, GpuVirtualAddressSpace>,
    /// Queue of available physical blocks to request for mapping
    physical_handles: VecDeque<CUmemGenericAllocationHandle>,
    ptr_bindings: PtrBindings,
}

impl GpuVirtualStorage {
    pub fn new(device_id: i32, granularity: usize, block_size: u64) -> Self {
        Self {
            device_id,
            mem_alignment: granularity,
            physical_block_size: block_size.next_multiple_of(granularity as u64),
            reservations: HashMap::new(),
            physical_handles: VecDeque::new(),
            ptr_bindings: PtrBindings::new(),
        }
    }

    fn allocate_physical_block(&mut self) -> Result<CUmemGenericAllocationHandle, IoError> {
        if let Some(block) = self.physical_handles.pop_front() {
            return Ok(block);
        };

        unsafe {
            let mut mem_handle: CUmemGenericAllocationHandle = 0;
            let mut prop: CUmemAllocationProp = std::mem::zeroed();

            prop.type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.requestedHandleTypes = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
            prop.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;
            prop.win32HandleMetaData = std::ptr::null_mut();
            prop.allocFlags.compressionType = 0;

            let result = cuMemCreate(&mut mem_handle, self.physical_block_size as usize, &prop, 0);

            match result {
                CUresult::CUDA_SUCCESS => Ok(mem_handle),
                CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
                    Err(IoError::BufferTooBig(self.physical_block_size as usize))
                }
                other => Err(IoError::Unknown(format!(
                    "CUDA alloc physical failed: {:?}",
                    other
                ))),
            }
        }
    }

    fn get_ptr(&self, handle: &StorageHandle) -> CUdeviceptr {
        self.reservations
            .get(&handle.id)
            .expect("Storage handle not found")
            .ptr()
    }

    fn is_addr_mapped(&self, handle: &StorageHandle) -> bool {
        self.reservations
            .get(&handle.id)
            .expect("Storage handle not found")
            .is_mapped()
    }

    fn set_access_permissions(
        &self,
        virtual_addr: CUdeviceptr,
        size: u64,
        device_id: i32,
    ) -> Result<(), IoError> {
        unsafe {
            let mut access_desc: CUmemAccessDesc = std::mem::zeroed();
            access_desc.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id = device_id;
            access_desc.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            let result = cuMemSetAccess(virtual_addr, size as usize, &access_desc, 1);

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
    fn granularity(&self) -> usize {
        self.mem_alignment
    }

    fn physical_block_size(&self) -> u64 {
        self.physical_block_size
    }

    fn reserve(&mut self, size: u64, start_addr: u64) -> Result<StorageHandle, IoError> {
        let aligned_size = size
            .saturating_sub(1)
            .next_multiple_of(self.physical_block_size);

        let blocks_needed = aligned_size.div_ceil(self.physical_block_size) as usize;

        let mut handles: Vec<CUmemGenericAllocationHandle> = Vec::with_capacity(blocks_needed);

        for _ in 0..blocks_needed {
            let block = self.allocate_physical_block()?;
            handles.push(block);
        }
        self.physical_handles.extend(handles);

        unsafe {
            let mut virtual_addr: CUdeviceptr = 0;

            // Note: It is not guaranteed that CUDA will reserve the address range at the address we request.
            // The fourth argument to [cuMemAddressReserve] acts like a 'hint' to tell the driver that we would like to
            // pre reserve memory starting at that point. It should be useful to expand virtual memory ranges when new memory is required.
            let result = cuMemAddressReserve(
                &mut virtual_addr,
                aligned_size as usize,
                self.mem_alignment,
                start_addr,
                0,
            );

            match result {
                CUresult::CUDA_SUCCESS => {
                    let id = StorageId::new();
                    let addr = GpuVirtualAddressSpace {
                        start_address: virtual_addr,
                        size: aligned_size,
                        handles: None,
                    };

                    self.reservations.insert(id, addr);

                    let handle = StorageHandle::new(id, StorageUtilization { size, offset: 0 });
                    Ok(handle)
                }
                // In theory, virtual address space reservations should not fail due to OOM, but I am not sure if there is an effective limit on the total virtual memory space size you can pre-reserve.
                CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
                    Err(IoError::BufferTooBig(aligned_size as usize))
                }
                other => Err(IoError::Unknown(format!(
                    "CUDA reserve failed: {:?}",
                    other
                ))),
            }
        }
    }

    fn release(&mut self, handle: StorageHandle) {
        let reservation = self
            .reservations
            .get(&handle.id)
            .expect("Storage handle not found");
        // Get reservation details before removing it
        let virtual_addr = reservation.ptr();
        let size = reservation.size();

        // If the reservation is mapped, unmap it first
        if reservation.is_mapped() {
            self.unmap(handle.id);
        }

        // Remove from reservations map
        self.reservations.remove(&handle.id);

        // Free the virtual address space
        unsafe {
            cuMemAddressFree(virtual_addr, size as usize);
        }
    }

    fn split_range(
        &mut self,
        handle: &mut StorageHandle,
        offset: u64,
    ) -> Result<StorageHandle, IoError> {
        let reservation = self
            .reservations
            .get(&handle.id)
            .expect("Storage handle not found");

        let effective_offset = offset
            .saturating_sub(1)
            .next_multiple_of(self.physical_block_size);
        let original_size = reservation.size();
        if reservation.is_mapped() || handle.offset() + effective_offset >= original_size {
            return Err(IoError::InvalidHandle);
        }

        let original_start = reservation.ptr();
        let original_size = reservation.size();

        let second_start = original_start + effective_offset + handle.offset();
        let second_size = original_size + handle.offset() - effective_offset;

        let second_id = StorageId::new();
        let second_reservation = GpuVirtualAddressSpace {
            start_address: second_start,
            size: second_size,
            handles: None,
        };

        let reservation_mut = self
            .reservations
            .get_mut(&handle.id)
            .expect("Storage handle not found");

        reservation_mut.size = effective_offset;
        self.reservations.insert(second_id, second_reservation);
        handle.utilization.size = effective_offset;

        let second_handle = StorageHandle::new(
            second_id,
            StorageUtilization {
                size: second_size,
                offset: 0,
            },
        );

        Ok(second_handle)
    }

    // Check whether two handles are adjacent in memory.
    fn are_adjacent(&self, first: &StorageHandle, second: &StorageHandle) -> bool {
        let first_reservation = match self.reservations.get(&first.id) {
            Some(res) => res,
            None => return false,
        };

        let second_reservation = match self.reservations.get(&second.id) {
            Some(res) => res,
            None => return false,
        };

        // Check if first handle ends exactly where second handle starts
        let first_end = first_reservation.ptr()
            + first
                .size()
                .saturating_sub(1)
                .next_multiple_of(self.physical_block_size);
        let second_start = second_reservation.ptr() + second.offset();

        first_end == second_start ||
        // Or if second handle ends exactly where first handle starts
        {
            let second_end = second_reservation.ptr() + second.size();
            let first_start = first_reservation.ptr() + first.offset();
            second_end == first_start
        }
    }

    fn merge(
        &mut self,
        first_handle: StorageHandle,
        second_handle: StorageHandle,
    ) -> Result<StorageHandle, IoError> {
        let first_reservation = self
            .reservations
            .remove(&first_handle.id)
            .ok_or_else(|| IoError::Unknown("First storage handle not found".to_string()))?;

        let second_reservation = self
            .reservations
            .remove(&second_handle.id)
            .ok_or_else(|| IoError::Unknown("Second storage handle not found".to_string()))?;

        if first_reservation.is_mapped() {
            return Err(IoError::Unknown(
                "Cannot merge mapped virtual address spaces. Unmap first handle first.".to_string(),
            ));
        }

        if second_reservation.is_mapped() {
            return Err(IoError::Unknown(
                "Cannot merge mapped virtual address spaces. Unmap second handle first."
                    .to_string(),
            ));
        }

        // If handles are contiguous there is no need to release and reallocate the virtual memory.
        if first_reservation.ptr() + first_reservation.size() == second_reservation.ptr() {
            let merged_reservation = GpuVirtualAddressSpace {
                start_address: first_reservation.start_address,
                size: first_reservation.size + second_reservation.size,
                handles: None,
            };

            let merged_id = StorageId::new();
            self.reservations.insert(merged_id, merged_reservation);
        }

        let total_size = first_handle.size() + second_handle.size();
        // Otherwise, we need to relase and reallocate.
        unsafe {
            cuMemAddressFree(first_reservation.ptr(), first_reservation.size() as usize);
            cuMemAddressFree(second_reservation.ptr(), second_reservation.size() as usize);
        }
        self.reserve(total_size, first_reservation.ptr())
    }

    fn expand(&mut self, handle: &mut StorageHandle, additional_size: u64) -> Result<(), IoError> {
        let aligned_size = handle
            .size()
            .saturating_sub(1u64)
            .next_multiple_of(self.physical_block_size)
            + additional_size;

        let ptr = self.get_ptr(handle);

        let new_handle = {
            let mut new_handle = self.reserve(additional_size, ptr + aligned_size)?.clone();

            if self.is_addr_mapped(handle) {
                self.map(&mut new_handle)?;
            };

            new_handle
        };

        let second_reservation = self
            .reservations
            .remove(&new_handle.id)
            .ok_or_else(|| IoError::Unknown("Second storage handle not found".to_string()))?;

        if ptr + aligned_size == second_reservation.ptr() {
            let first_reservation_mut = self
                .reservations
                .get_mut(&handle.id)
                .ok_or_else(|| IoError::Unknown("Storage handle not found".to_string()))?;
            // Reserving contiguous memory succeeded, can merge inplace
            first_reservation_mut.size = aligned_size;
        } else {
            unsafe {
                cuMemAddressFree(second_reservation.ptr(), second_reservation.size() as usize);
            }
            return Err(IoError::InvalidHandle);
        }
        Ok(())
    }

    /// Maps a prereserved memory range to a number of preallocated physical blocks.
    fn map(&mut self, handle: &mut StorageHandle) -> Result<(), IoError> {
        let aligned_size = handle
            .size()
            .saturating_sub(1)
            .next_multiple_of(self.mem_alignment as u64);
        let blocks_needed = aligned_size.div_ceil(self.physical_block_size) as usize;
        let mut handles: Vec<CUmemGenericAllocationHandle> = Vec::with_capacity(blocks_needed);

        for _ in 0..blocks_needed {
            let block = self.allocate_physical_block()?;
            handles.push(block);
        }

        let space = self
            .reservations
            .get(&handle.id)
            .expect("Storage handle not found");

        assert!(
            !space.is_mapped(),
            "Requested to map an aleady mapped virtual address space. This is invalid. First unmap the handle."
        );

        // Map each block
        let mut mapped_ranges = Vec::with_capacity(blocks_needed);
        let mut current_addr = space.ptr();
        let size = space.size();

        for handle in handles.iter() {
            unsafe {
                let result = cuMemMap(
                    current_addr,
                    self.physical_block_size as usize,
                    0,
                    *handle,
                    0,
                );
                match result {
                    CUresult::CUDA_SUCCESS => {
                        if let Err(e) = self.set_access_permissions(
                            current_addr,
                            self.physical_block_size,
                            self.device_id,
                        ) {
                            // Rollback all successful mappings.
                            for &addr in &mapped_ranges {
                                cuMemUnmap(addr, self.physical_block_size as usize);
                            }
                            // Return all handles.
                            for h in handles {
                                self.physical_handles.push_back(h);
                            }
                            return Err(e);
                        }
                        mapped_ranges.push(current_addr);
                        current_addr += self.physical_block_size;
                    }
                    other => {
                        for &addr in &mapped_ranges {
                            cuMemUnmap(addr, size as usize);
                        }

                        for h in handles {
                            self.physical_handles.push_back(h);
                        }
                        return Err(IoError::Unknown(format!("CUDA map failed: {:?}", other)));
                    }
                }
            }
        }

        let space_mut = self
            .reservations
            .get_mut(&handle.id)
            .expect("Storage handle not found");
        space_mut.handles = Some(handles);
        Ok(())
    }

    fn unmap(&mut self, id: StorageId) {
        if let Some(mapping) = self.reservations.get_mut(&id) {
            assert!(
                mapping.is_mapped(),
                "Requested to unmap an already unmapped virtual address space. This is invalid. First map the handle to physical memory."
            );

            let mut current_addr = mapping.ptr();
            let num_blocks = mapping.size().div_ceil(self.physical_block_size);
            for _block in 0..num_blocks {
                unsafe {
                    cuMemUnmap(current_addr, self.physical_block_size as usize);
                }
                current_addr += self.physical_block_size;
            }

            // Return all handles to the pool.
            let handles = mapping.handles.take().unwrap();
            self.physical_handles.extend(handles);
        }
    }

    fn cleanup(&mut self) {
        unsafe {
            for (_, mut mapping) in self.reservations.drain() {
                if let Some(handles) = mapping.handles.take() {
                    cuMemUnmap(mapping.ptr(), mapping.size() as usize);
                    for handle in handles {
                        cuMemRelease(handle);
                    }
                }

                cuMemAddressFree(mapping.ptr(), mapping.size() as usize);
            }

            for handle in self.physical_handles.drain(..) {
                cuMemRelease(handle);
            }
        }
    }

    /// Completely defragments the virtual address space, combining all unmapped ranges into a single one and returning the resulting handle.
    fn defragment(&mut self) -> Option<StorageHandle> {
        // Collect all unmapped reservations
        let mut reservations: Vec<(StorageId, CUdeviceptr, u64)> = self
            .reservations
            .iter()
            .filter(|(_, space)| !space.is_mapped())
            .map(|(id, space)| (*id, space.start_address, space.size))
            .collect();

        if reservations.is_empty() {
            return None;
        }

        reservations.sort_by_key(|(_, addr, _)| *addr);

        // 2. Attempt to merge contiguous
        let mut i = 0;
        while i + 1 < reservations.len() {
            let (id_a, addr_a, size_a) = reservations[i];
            let (id_b, addr_b, size_b) = reservations[i + 1];

            let can_merge = addr_a + size_a == addr_b;

            if can_merge {
                let mut a = self.reservations.remove(&id_a).unwrap();
                let b = self.reservations.remove(&id_b).unwrap();

                a.size += b.size;
                let merged_id = StorageId::new();
                self.reservations.insert(merged_id, a);

                reservations[i] = (merged_id, addr_a, size_a + size_b);
                reservations.remove(i + 1);
            } else {
                i += 1;
            }
        }

        // If all unmapped handles were sucessfully merged, return the resulting handle.
        if reservations.len() == 1 {
            let (id, _, size) = reservations[0];
            return Some(StorageHandle {
                id,
                utilization: StorageUtilization { offset: 0, size },
            });
        }

        // If there are remaining handles, free them and reallocate them
        let total_size: u64 = reservations.iter().map(|(_, _, size)| size).sum();

        for (id, _, size) in &reservations {
            if let Some(space) = self.reservations.remove(id) {
                unsafe {
                    cuMemAddressFree(space.ptr(), *size as usize);
                }
            }
        }

        unsafe {
            let mut virtual_addr: CUdeviceptr = 0;
            let result = cuMemAddressReserve(
                &mut virtual_addr,
                total_size as usize,
                self.mem_alignment,
                0,
                0,
            );

            if result == CUresult::CUDA_SUCCESS {
                let new_id = StorageId::new();
                let new_space = GpuVirtualAddressSpace {
                    start_address: virtual_addr,
                    size: total_size,
                    handles: None,
                };
                self.reservations.insert(new_id, new_space);

                return Some(StorageHandle {
                    id: new_id,
                    utilization: StorageUtilization {
                        offset: 0,
                        size: total_size,
                    },
                });
            }
        }

        None
    }
}

impl ComputeStorage for GpuVirtualStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.physical_block_size as usize
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let reservation = self
            .reservations
            .get(&handle.id)
            .expect("Storage handle not found");

        assert!(
            reservation.is_mapped(),
            "Attempted to get an unmapped virtual address range. This is invalid. First map the handle to be able to use it"
        );

        let ptr = reservation.ptr();
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
        let mut handle = self.reserve(size, 0)?;
        self.map(&mut handle)?;
        Ok(handle)
    }

    fn dealloc(&mut self, id: StorageId) {
        self.unmap(id);
    }

    fn flush(&mut self) {
        self.cleanup();
    }
}

impl Drop for GpuVirtualStorage {
    fn drop(&mut self) {
        self.cleanup();
    }
}
unsafe impl Send for GpuVirtualStorage {}




pub fn get_minimum_granularity(device: CUdevice) -> usize {
        unsafe {
            let mut granularity = 0;
            let mut prop: CUmemAllocationProp = std::mem::zeroed();
            prop.type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;

            let result = cuMemGetAllocationGranularity(
                &mut granularity,
                &prop,
                CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            );

            match result {
                CUresult::CUDA_SUCCESS => granularity,
                _ => 64 * 1024, // Fallback a 64KB
            }
        }
    }

#[cfg(test)]
mod virtual_mem_tests {
    use super::*;
    use cubecl_runtime::storage::{ComputeStorage, VirtualStorage};
    use cudarc::driver::{result, sys::*};

    fn setup_cuda_context() -> (CUdevice, CUcontext) {
        result::init().unwrap();
        let device = result::device::get(0).unwrap();
        let ctx = unsafe {
            let ctx = result::primary_ctx::retain(device).unwrap();
            result::ctx::set_current(ctx).unwrap();
            ctx
        };
        (device, ctx)
    }



    #[test]
    fn test_reserve_and_release() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = (granularity * 2) as u64;

        let mut storage = GpuVirtualStorage::new(0, granularity, block_size);

        let size = block_size * 2;
        let handle = storage.reserve(size, 0).unwrap();

        assert_eq!(handle.size(), size as u64);
        assert_eq!(handle.offset(), 0);

        assert!(!storage.is_addr_mapped(&handle));
        storage.release(handle);

        assert!(storage.reservations.is_empty());
    }

    #[test]
    fn test_map_and_unmap() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = (granularity * 2) as u64;

        let mut storage = GpuVirtualStorage::new(0, granularity, block_size);

        let size = block_size;
        let mut handle = storage.reserve(size, 0).unwrap();

        assert!(!storage.is_addr_mapped(&handle));

        storage.map(&mut handle).unwrap();
        assert!(storage.is_addr_mapped(&handle));

        storage.unmap(handle.id);
        assert!(!storage.is_addr_mapped(&handle));

        storage.release(handle);
    }

    #[test]
    fn test_alloc_and_dealloc() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = (granularity * 2) as u64;

        let mut storage = GpuVirtualStorage::new(0, granularity, block_size);

        assert_eq!(storage.physical_handles.len(), 0);

        let size = block_size as u64;
        let handle = storage.alloc(size).unwrap();

        assert_eq!(handle.size(), size);
        assert!(storage.is_addr_mapped(&handle));

        let resource = storage.get(&handle);
        assert_eq!(resource.size, size);

        storage.dealloc(handle.id);
        assert!(!storage.is_addr_mapped(&handle));

        let expected_blocks = size.div_ceil(block_size);
        assert_eq!(storage.physical_handles.len(), expected_blocks as usize);

        let handle2 = storage.alloc(size).unwrap();

        assert_eq!(storage.physical_handles.len(), 0);

        storage.dealloc(handle2.id);
    }

    #[test]
    fn test_split_range() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = granularity as u64;

        let mut storage = GpuVirtualStorage::new(0, granularity, block_size);

        let total_size = block_size * 3;

        let mut handle = storage.reserve(total_size, 0).unwrap();

        let split_offset = block_size * 2;
        let second_handle = storage.split_range(&mut handle, split_offset).unwrap();

        assert_eq!(handle.size(), split_offset as u64);
        println!("Tamaño del primer handle: {:?}", handle.size());
        assert_eq!(second_handle.size(), (total_size - split_offset) as u64);
        assert_eq!(second_handle.offset(), 0);

        storage.release(handle);
        storage.release(second_handle);
    }

    #[test]
    fn test_are_adjacent() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = (granularity * 2) as u64;

        let mut storage = GpuVirtualStorage::new(0, granularity, block_size);
        let total_size = block_size * 4;
        let mut first_handle = storage.reserve(total_size, 0).unwrap();

        let split_offset = block_size * 2;
        let second_handle = storage
            .split_range(&mut first_handle, split_offset)
            .unwrap();
        assert!(storage.are_adjacent(&first_handle, &second_handle));
        assert!(storage.are_adjacent(&second_handle, &first_handle));
        storage.release(first_handle);
        storage.release(second_handle);
    }

    #[test]
    fn test_granularity_and_block_size() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = (granularity * 3) as u64;

        let storage = GpuVirtualStorage::new(0, granularity, block_size);

        assert_eq!(storage.granularity(), granularity);

        assert_eq!(storage.physical_block_size(), block_size);
        assert_eq!(storage.alignment(), block_size as usize);
    }

  

    #[test]
    fn test_defragment_contiguous_ranges() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = (granularity * 2) as u64;

        let mut storage = GpuVirtualStorage::new(0, granularity, block_size);


        let total_size = block_size * 6;
        let mut handle1 = storage.reserve(total_size, 0).unwrap();


        let split_size = block_size * 2;
        let mut handle2 = storage.split_range(&mut handle1, split_size).unwrap();
        let handle3 = storage.split_range(&mut handle2, split_size).unwrap();


        assert_eq!(storage.reservations.len(), 3);
        assert_eq!(handle1.size(), split_size as u64);
        assert_eq!(handle2.size(), split_size as u64);
        assert_eq!(handle3.size(), split_size as u64);

        assert!(storage.are_adjacent(&handle1, &handle2));
        assert!(storage.are_adjacent(&handle2, &handle3));

        // Defragment
        let defrag_result = storage.defragment();

        assert!(defrag_result.is_some(), "Defragmentation should succeed with contiguous ranges");
        let merged_handle = defrag_result.unwrap();


        assert_eq!(merged_handle.size(), total_size as u64);
        assert_eq!(merged_handle.offset(), 0);


        assert_eq!(storage.reservations.len(), 1);

        storage.release(merged_handle);
    }

    #[test]
    fn test_defragment_scattered_ranges() {
        let (_device, _ctx) = setup_cuda_context();
        let device = result::device::get(0).unwrap();
        let granularity = get_minimum_granularity(device);
        let block_size = (granularity * 2) as u64;

        let mut storage = GpuVirtualStorage::new(0, granularity, block_size);


        let size1 = block_size * 2;
        let size2 = block_size * 3;
        let size3 = block_size * 1;

        let handle1 = storage.reserve(size1, 0).unwrap();
        let mut handle2 = storage.reserve(size2, 0).unwrap();
        let handle3 = storage.reserve(size3, 0).unwrap();

        storage.map(&mut handle2).unwrap();

        assert_eq!(storage.reservations.len(), 3);

        assert!(!storage.is_addr_mapped(&handle1));
        assert!(storage.is_addr_mapped(&handle2));
        assert!(!storage.is_addr_mapped(&handle3));


        let defrag_result = storage.defragment();


        if defrag_result.is_some() {
            let merged_handle = defrag_result.unwrap();
            let expected_size = (size1 + size3) as u64;

            assert_eq!(merged_handle.size(), expected_size);
            assert_eq!(merged_handle.offset(), 0);


            assert!(storage.reservations.contains_key(&handle2.id));
            assert!(storage.is_addr_mapped(&handle2));


            storage.unmap(handle2.id);
            storage.release(handle2);
            storage.release(merged_handle);
        } else {

            storage.unmap(handle2.id);
            storage.release(handle1);
            storage.release(handle2);
            storage.release(handle3);

            panic!("Storage should have defragmented propertly!");
        }
    }
}
