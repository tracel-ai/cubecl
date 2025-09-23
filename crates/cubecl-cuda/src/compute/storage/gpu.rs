use crate::compute::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{
    ComputeStorage, StorageHandle, StorageId, StorageUtilization, VirtualStorage,
};
use cudarc::driver::DriverError;
use cudarc::driver::sys::*;
use std::collections::BTreeMap;
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
    handles: BTreeMap<CUdeviceptr, Option<CUmemGenericAllocationHandle>>,
}

impl GpuVirtualAddressSpace {
    fn ptr(&self) -> CUdeviceptr {
        self.start_address
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn with_block_size(start_address: CUdeviceptr, size: u64, block_size: u64) -> Self {
        let num_handles = size.div_ceil(block_size);
        let mut handles = BTreeMap::new();
        let mut current_addr = start_address;

        for i in 0u64..num_handles {
            handles.insert(current_addr, None);
            current_addr += i * block_size;
        }

        Self {
            start_address,
            size,
            handles,
        }
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
    pub fn new(device_id: i32, granularity: usize, physical_block_size: u64) -> Self {
        let physical_block_size = physical_block_size
            .saturating_sub(1)
            .next_multiple_of(granularity as u64);
        Self {
            device_id,
            mem_alignment: granularity,
            physical_block_size,
            reservations: HashMap::new(),
            physical_handles: VecDeque::new(),
            ptr_bindings: PtrBindings::new(),
        }
    }

    fn allocate_physical_block(&mut self) -> Result<CUmemGenericAllocationHandle, IoError> {
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

    fn allocate(&mut self, size: u64) -> Result<(), IoError> {
        let total_size = size
            .saturating_sub(1)
            .next_multiple_of(self.physical_block_size);
        let num_blocks = total_size.div_ceil(self.physical_block_size);
        for _ in 0..num_blocks {
            let handle = self.allocate_physical_block()?;
            self.physical_handles.push_back(handle);
        }
        Ok(())
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
                    let addr = GpuVirtualAddressSpace::with_block_size(
                        virtual_addr,
                        aligned_size,
                        self.physical_block_size,
                    );

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

    fn release(&mut self, id: StorageId) {
        let reservation = self
            .reservations
            .get(&id)
            .expect("Storage handle not found");

        // Get reservation details before removing it
        let virtual_addr = reservation.ptr();
        let size = reservation.size();
        self.unmap(id, 0, reservation.size());

        // Remove from reservations map
        self.reservations.remove(&id);

        // Free the virtual address space
        unsafe {
            cuMemAddressFree(virtual_addr, size as usize);
        }
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
        let first_end = first_reservation.ptr() + first_reservation.size();

        let second_start = second_reservation.ptr();

        first_end == second_start ||
        // Or if second handle ends exactly where first handle starts
        {
            let second_end = second_reservation.ptr() + second_reservation.size();
            let first_start = first_reservation.ptr();
            second_end == first_start
        }
    }

    /// Maps a prereserved memory range to a number of preallocated physical blocks.
    fn map(&mut self, id: StorageId, offset: u64, size: u64) -> Result<StorageHandle, IoError> {
        let aligned_offset = offset
            .saturating_sub(1)
            .next_multiple_of(self.physical_block_size);

        let aligned_size = size
            .saturating_sub(1)
            .next_multiple_of(self.physical_block_size);

        let space_mut = self
            .reservations
            .get_mut(&id)
            .expect("Storage handle not found");

            // Map each block
        let size = space_mut.size();

        if aligned_offset + aligned_size > size {
            return Err(IoError::InvalidHandle);
        }



        for (addr, mapped_handle) in space_mut
            .handles
            .range_mut(aligned_offset..aligned_offset + aligned_size)
        {
            if mapped_handle.is_some() {
                // Return error: already mapped.
                return Err(IoError::InvalidHandle);
            }

            let handle = match self.physical_handles.pop_front() {
                Some(h) => h,
                None => return Err(IoError::Unknown("No free physical handles".into())),
            };

            unsafe {
                let result = cuMemMap(*addr, self.physical_block_size as usize, 0, handle, 0);
                match result {
                    CUresult::CUDA_SUCCESS => {

                            *mapped_handle = Some(handle);

                    }
                    other => {
                        return Err(IoError::Unknown(format!("CUDA map failed: {:?}", other)));
                    }
                }
            }
        }

         let space = self
            .reservations
            .get(&id)
            .expect("Storage handle not found");

        for (addr, mapped_handle) in space
            .handles
            .range(aligned_offset..aligned_offset + aligned_size)
        {

            self.set_access_permissions(*addr, self.physical_block_size, self.device_id)?;

        }

        let handle = StorageHandle::new(id, StorageUtilization { offset, size });
        Ok(handle)
    }

    fn unmap(&mut self, id: StorageId, offset: u64, size: u64) {
        let aligned_offset = offset
            .saturating_sub(1)
            .next_multiple_of(self.physical_block_size);
        let aligned_size = size
            .saturating_sub(1)
            .next_multiple_of(self.physical_block_size);

        if let Some(mapping) = self.reservations.get_mut(&id) {
            for (addr, mapped_handle) in mapping
                .handles
                .range_mut(aligned_offset..aligned_offset + aligned_size)
            {
                if let Some(handle) = mapped_handle.take() {
                    unsafe {
                        cuMemUnmap(*addr, self.physical_block_size as usize);
                    }

                    // Return all handles to the pool.
                    self.physical_handles.push_back(handle);
                }
            }
        }
    }

    fn cleanup(&mut self) {
        for (id, mut reservation) in self.reservations.drain() {
            // Get reservation details before removing it
            let virtual_addr = reservation.ptr();
            let size = reservation.size();

            for (addr, mapped_handle) in reservation.handles.range_mut(0..size) {
                if let Some(handle) = mapped_handle.take() {
                    unsafe {
                        cuMemUnmap(*addr, self.physical_block_size as usize);
                    }

                    // Return all handles to the pool.
                    self.physical_handles.push_back(handle);
                }
            }

            // Free the virtual address space
            unsafe {
                cuMemAddressFree(virtual_addr, size as usize);
            }
        }

        for handle in self.physical_handles.drain(..) {
            unsafe {
                cuMemRelease(handle);
            }
        }
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
        self.map(handle.id, 0, size)?;
        Ok(handle)
    }

    fn are_contiguous(&self, handle1: &StorageHandle, handle2: &StorageHandle) -> bool {
        self.are_adjacent(handle1, handle2)
    }

    fn dealloc(&mut self, id: StorageId) {
        if let Some(reservation) = self.reservations.get(&id) {
            self.unmap(id, 0, reservation.size());
        }
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
}
