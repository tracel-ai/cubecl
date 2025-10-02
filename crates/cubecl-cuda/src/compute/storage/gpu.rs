use crate::compute::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{
    ComputeStorage, PhysicalStorageHandle, PhysicalStorageId, StorageHandle, StorageId,
    StorageUtilization, VirtualStorage,
};
use cudarc::driver::DriverError;
use cudarc::driver::sys::*;
use std::collections::HashMap;

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

/// A Gpu Physical Memory block of a specific size.
#[allow(dead_code)]
struct GpuMemoryBlock {
    handle: CUmemGenericAllocationHandle,
    size: u64,
    mapped: bool,
}
#[allow(dead_code)]
impl GpuMemoryBlock {
    /// Getter for internal handle (pointer to physical device memory)
    fn handle(&self) -> CUmemGenericAllocationHandle {
        self.handle
    }

    /// Getter for size.
    fn size(&self) -> u64 {
        self.size
    }

    /// Set the block to mapped state
    fn set_mapped(&mut self, mapped: bool) {
        self.mapped = mapped;
    }

    /// Constructor
    fn new(handle: CUmemGenericAllocationHandle, size: u64) -> Self {
        Self {
            handle,
            size,
            mapped: false,
        }
    }
}

/// A Gpu virtual address space of a specific size
#[allow(dead_code)]
struct GpuVirtualAddressSpace {
    start_address: CUdeviceptr,
    size: u64,
}
#[allow(dead_code)]
impl GpuVirtualAddressSpace {
    /// Getter for the start of this space
    fn ptr(&self) -> CUdeviceptr {
        self.start_address
    }

    /// Getter for the size
    fn size(&self) -> u64 {
        self.size
    }

    /// Constructor
    fn new(start_address: CUdeviceptr, size: u64) -> Self {
        Self {
            start_address,
            size,
        }
    }
}

/// Memory allocation mode.
#[derive(Default)]
#[allow(dead_code)]
pub enum GpuAllocationMode {
    /// Standard allocation mode disables virtual memory support
    #[default]
    Standard = 0,
    /// Virtual memory is enabled
    Virtual = 1,
}

/// A CUDA buffer storage with support for virtual memory.
/// Maybe you could consider refactoring by merging this struct with the GpuStorage struct, so that a single GpuStorage is preserved. This storage type implements both the [`VirtualStorage`]and the [`ComputeStorage`] trait, which allows it to manage both types of allocations depending on the [`mode`].
#[allow(dead_code)]
pub struct GpuVirtualStorage {
    /// Id of the device where the virtual memory is going to be allocated with this storage.
    device_id: i32,
    /// Minimum allocation granularity of the target device. Can be set to other value if allocation mode is not virtual memory.
    mem_alignment: usize,
    /// Reserved virtual memory ranges. Can be either mapped or unmapped.
    virtual_memory: HashMap<StorageId, GpuVirtualAddressSpace>,
    /// Map of  physical blocks
    physical_memory: HashMap<PhysicalStorageId, GpuMemoryBlock>,
    /// Memory: I want this storage type to also support standard allocation mode, so that both standard pools and virtual pools can use it.
    memory: HashMap<StorageId, cudarc::driver::sys::CUdeviceptr>,
    /// Deallocations
    deallocations: Vec<StorageId>,
    /// Stream
    stream: cudarc::driver::sys::CUstream,
    /// Ptr bindings
    ptr_bindings: PtrBindings,
    /// Mode
    alloc_mode: GpuAllocationMode,
}

#[allow(dead_code)]
impl GpuVirtualStorage {
    pub fn new(
        device_id: i32,
        stream: cudarc::driver::sys::CUstream,
        granularity: usize,
        alloc_mode: GpuAllocationMode,
    ) -> Self {
        Self {
            device_id,
            mem_alignment: granularity,
            virtual_memory: HashMap::new(),
            physical_memory: HashMap::new(),
            ptr_bindings: PtrBindings::new(),
            memory: HashMap::new(),
            deallocations: Vec::new(),
            stream,
            alloc_mode,
        }
    }

    /// Utility to allocate a single physical block of a target size.
    /// Assumes size is aligned to [`self.granularity()`]
    fn allocate_physical_block(&mut self, size: u64) -> Result<GpuMemoryBlock, IoError> {
        assert_eq!(
            size % self.granularity() as u64,
            0,
            "For virtual memory allocations, size must be aligned to self.granularity()"
        );

        unsafe {
            let mut mem_handle: CUmemGenericAllocationHandle = 0;
            let mut prop: CUmemAllocationProp = std::mem::zeroed();

            prop.type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.requestedHandleTypes = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
            prop.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;
            prop.win32HandleMetaData = std::ptr::null_mut();
            prop.allocFlags.compressionType = 0;

            let result = cuMemCreate(&mut mem_handle, size as usize, &prop, 0);

            match result {
                CUresult::CUDA_SUCCESS => {
                    let block = GpuMemoryBlock::new(mem_handle, size);
                    Ok(block)
                }
                CUresult::CUDA_ERROR_OUT_OF_MEMORY => Err(IoError::BufferTooBig(size as usize)),
                other => Err(IoError::Unknown(format!(
                    "CUDA alloc physical failed: {:?}",
                    other
                ))),
            }
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

    /// Sets the access permissions for the target device on an allocated and mapped virtual address space.
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

/// Virtual storage trait implementation
impl VirtualStorage for GpuVirtualStorage {
    /// Minimum allocation granularity for the target device.
    fn granularity(&self) -> usize {
        self.mem_alignment
    }

    /// Whether virtual memory is enabled (depends on allocation mode)
    fn is_virtual_mem_enabled(&self) -> bool {
        matches!(self.alloc_mode, GpuAllocationMode::Virtual)
    }

    /// Allocates a physical memory block of the target size, ensuring the size is aligned to [`self.granularity()`].
    /// Returns a handle containing the size and status of the allocation (unmapped).
    /// The handle will be set as mapped on the [`map()`] method.
    fn allocate(&mut self, size: u64) -> Result<PhysicalStorageHandle, IoError> {
        if !self.is_virtual_mem_enabled() {
            return Err(IoError::Unknown("Virtual memory is disabled!".to_string()));
        }

        let total_size = size.next_multiple_of(self.mem_alignment as u64);
        let block = self.allocate_physical_block(total_size)?;

        let id = PhysicalStorageId::new();
        let phys = PhysicalStorageHandle::new(
            id,
            StorageUtilization {
                size,
                offset: 0, // Offset is always 0 for physical handles, as we cannot partition them for now.
            },
        );
        self.physical_memory.insert(id, block);

        Ok(phys)
    }

    /// Checks if two address spaces are contiguous in memory (the first one ends where the second one starts).
    /// This is useful to perform defragmentation.
    fn are_aligned(&self, lhs: &StorageId, rhs: &StorageId) -> bool {
        if let Some(a) = self.virtual_memory.get(lhs)
            && let Some(b) = self.virtual_memory.get(rhs)
        {
            return a.ptr() + a.size() == b.ptr();
        };
        false
    }

    /// Reserves a virtual address space of the target size.
    /// The parameter `start_addr` is a hint to tell CUDA where do we want the allocation to start.
    /// However, in practice the CUDA documentations says it is not guaranteed for the allocation to start where we want it to.
    /// Returns a storage handle pointing to the reserved virtual address space.
    fn reserve(
        &mut self,
        size: u64,
        start_addr: Option<StorageId>,
    ) -> Result<StorageHandle, IoError> {
        if !self.is_virtual_mem_enabled() {
            return Err(IoError::Unknown("Virtual memory is disabled!".to_string()));
        }

        let aligned_size = size.next_multiple_of(self.mem_alignment as u64);

        let addr = if let Some(prev) = start_addr
            && let Some(space) = self.virtual_memory.get(&prev)
        {
            space.ptr() + space.size()
        } else {
            0 // Zero will tell CUDA  to autotune
        };

        unsafe {
            let mut virtual_addr: CUdeviceptr = 0;

            // Note: It is not guaranteed that CUDA will reserve the address range at the address we request.
            // The fourth argument to [cuMemAddressReserve] acts like a 'hint' to tell the driver that we would like to
            // pre reserve memory starting at that point. It should be useful to expand virtual memory ranges when new memory is required.
            let result = cuMemAddressReserve(
                &mut virtual_addr,
                aligned_size as usize,
                self.mem_alignment,
                addr,
                0,
            );

            match result {
                CUresult::CUDA_SUCCESS => {
                    let id = StorageId::new();
                    let addr = GpuVirtualAddressSpace::new(virtual_addr, aligned_size);

                    self.virtual_memory.insert(id, addr);

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

    /// Unmaps and frees a virtual address space, removing from the
    // Virtual memory tracking structure.
    /// Acts like a no op if it does not find the reservation.
    fn free(&mut self, id: StorageId) {
        if let Some(reservation) = self.virtual_memory.remove(&id) {
            // Get reservation details before removing it
            let virtual_addr = reservation.ptr();
            let size = reservation.size();

            unsafe {
                cuMemUnmap(virtual_addr, size as usize);
            }

            // Free the virtual address space
            unsafe {
                cuMemAddressFree(virtual_addr, size as usize);
            }
        }
    }

    /// Frees a physical memory chunk by returning it to the driver.
    fn release(&mut self, id: PhysicalStorageId) {
        if let Some(handle) = self.physical_memory.remove(&id) {
            // Runtime check to ensure we do not attempt to free mapped memory
            assert!(!handle.mapped, "Cannot release a mapped handle!");
            unsafe {
                cuMemRelease(handle.handle());
            }
        }
    }

    /// Maps a portion of a virtual address space to a physical block.
    /// The portion will start at the next aligned offset and will have
    /// A size matching the next aligned size of the provided physical memory handle.
    fn map(
        &mut self,
        id: StorageId,
        offset: u64,
        physical: &mut PhysicalStorageHandle,
    ) -> Result<StorageHandle, IoError> {
        if !self.is_virtual_mem_enabled() {
            return Err(IoError::Unknown("Virtual memory is disabled!".to_string()));
        }

        let aligned_offset = offset.next_multiple_of(self.mem_alignment as u64);
        let space = self
            .virtual_memory
            .get(&id)
            .expect("Storage handle not found");

        let ph = self
            .physical_memory
            .get(&physical.id())
            .expect("Storage handle not found");

        // Map each block
        let size = space.size();
        let ph_size = ph.size();

        if (aligned_offset + ph_size) > size {
            return Err(IoError::InvalidHandle);
        }
        let addr = space.start_address + aligned_offset;
        unsafe {
            if let Err(e) = cuMemMap(addr, ph_size as usize, 0, ph.handle(), 0).result() {
                return Err(IoError::Unknown(format!("CUDA map failed: {:?}", e)));
            }
        }

        self.set_access_permissions(addr, ph_size, self.device_id)?;
        physical.set_mapped(true);

        let ph_mut = self
            .physical_memory
            .get_mut(&physical.id())
            .expect("Storage handle not found");
        ph_mut.set_mapped(true);
        let handle = StorageHandle::new(
            id,
            StorageUtilization {
                offset,
                size: ph_size,
            },
        );
        Ok(handle)
    }

    /// Unmaps a mapped virtual memory portion from a physical handle.
    /// Both the address space and the handle should be valid. However, it does not check whether this two structures are actually mapped. Therefore, it is the responsibility of the memory pool to keep track of actual mappings from physical handles to virtual address spaces.
    fn unmap(&mut self, id: StorageId, offset: u64, physical: &mut PhysicalStorageHandle) {
        // Offset should be aligned at this level, however there is no issue in enforcing it by explicit alignment.
        let aligned_offset = offset.next_multiple_of(self.mem_alignment as u64);

        if let Some(ph) = self.physical_memory.get_mut(&physical.id()) {
            let aligned_size = ph.size();

            if let Some(mapping) = self.virtual_memory.get(&id) {
                let addr = mapping.ptr() + aligned_offset;
                unsafe {
                    cuMemUnmap(addr, aligned_size as usize);
                }

                physical.set_mapped(false);
                ph.set_mapped(false);
            }
        }
    }
}

impl ComputeStorage for GpuVirtualStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        // It is critical to manage this properly.
        let ptr = match self.alloc_mode {
            GpuAllocationMode::Standard => self
                .memory
                .get(&handle.id)
                .expect("Storage handle not found"),
            GpuAllocationMode::Virtual => &self
                .virtual_memory
                .get(&handle.id)
                .expect("Storage handle not found")
                .ptr(),
        };

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

/// When dropped releases all resources.
impl Drop for GpuVirtualStorage {
    fn drop(&mut self) {
        self.flush();

        for (_, reservation) in self.virtual_memory.drain() {
            // Get reservation details before removing it
            let virtual_addr = reservation.ptr();
            let size = reservation.size();

            unsafe {
                cuMemUnmap(virtual_addr, size as usize);
            }

            // Free the virtual address space
            unsafe {
                cuMemAddressFree(virtual_addr, size as usize);
            }
        }

        for (_, handle) in self.physical_memory.drain() {
            unsafe {
                cuMemRelease(handle.handle());
            }
        }
    }
}

unsafe impl Send for GpuVirtualStorage {}
impl cubecl_runtime::storage::VirtualStorage for GpuStorage {}

pub fn get_minimum_granularity(device: CUdevice) -> usize {
    unsafe {
        let mut granularity = 0;
        let mut prop: CUmemAllocationProp = std::mem::zeroed();
        prop.type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;

        let result = cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        );

        match result {
            CUresult::CUDA_SUCCESS => granularity,
            _ => 0,
        }
    }
}

/// TODO: MIGRATE THIS TO A SEPARATED MACRO that is generic over Storage
///
/// Creates a new GpuVirtualStorage for testing
#[cfg(test)]
fn setup_cuda_storage() -> GpuVirtualStorage {
    cudarc::driver::result::init().unwrap();
    let device = cudarc::driver::result::device::get(0).unwrap();

    // Context must be initialized before creating a stream.
    let _ctx = unsafe {
        let ctx = cudarc::driver::result::primary_ctx::retain(device).unwrap();
        cudarc::driver::result::ctx::set_current(ctx).unwrap();
        ctx
    };

    let stream = cudarc::driver::result::stream::create(
        cudarc::driver::result::stream::StreamKind::NonBlocking,
    )
    .expect("Can create a new stream.");
    let granularity = get_minimum_granularity(device);
    GpuVirtualStorage::new(device, stream, granularity, GpuAllocationMode::Virtual)
}

#[cfg(test)]
mod virtual_storage_tests {
    use super::*;
    use cubecl_runtime::storage::VirtualStorage;

    #[test]
    /// Validates a simple physical memory allocation pattern.
    fn test_physical_memory_allocation() {
        let mut storage = setup_cuda_storage();
        let granularity = storage.granularity();
        let size = granularity as u64 * 4; // 4 times the minimum granularity

        let handle_result = storage.allocate(size);
        assert!(handle_result.is_ok());

        let handle = handle_result.unwrap();
        assert_eq!(handle.size(), size);
        assert!(!handle.is_mapped());

        // Verify the physical memory block was stored
        assert!(storage.physical_memory.contains_key(&handle.id()));

        // Clean up
        storage.release(handle.id());
        assert!(!storage.physical_memory.contains_key(&handle.id()));
    }

    #[test]
    /// Validates a simple virtual address space reservation
    fn test_virtual_address_space_reservation() {
        let mut storage = setup_cuda_storage();
        let granularity = storage.granularity();

        // Will reserve an address space of 8 times the granularity.
        let size = granularity as u64 * 8;
        let handle_result = storage.reserve(size, None);
        assert!(handle_result.is_ok());

        let handle = handle_result.unwrap();
        assert_eq!(handle.size(), size);
        assert_eq!(handle.offset(), 0);

        // Verify the virtual address space was stored
        assert!(storage.virtual_memory.contains_key(&handle.id));

        let virtual_space = storage.virtual_memory.get(&handle.id).unwrap();
        assert!(virtual_space.ptr() != 0); // Should have a valid pointer
        assert_eq!(virtual_space.size(), size);

        // Clean up
        storage.free(handle.id);
        assert!(!storage.virtual_memory.contains_key(&handle.id));
    }

    #[test]
    /// Validates a simple virtual address space mapping
    fn test_memory_mapping() {
        let mut storage = setup_cuda_storage();
        let granularity = storage.granularity();
        let size = granularity as u64 * 2;

        // Allocate physical memory
        let mut physical_handle = storage.allocate(size).unwrap();
        assert!(!physical_handle.is_mapped());

        // Reserve virtual address space
        let virtual_handle = storage.reserve(size, None).unwrap();

        // Map physical to virtual
        let map_result = storage.map(virtual_handle.id, 0, &mut physical_handle);
        assert!(map_result.is_ok());

        let mapped_handle = map_result.unwrap();
        assert_eq!(mapped_handle.size(), size);
        assert_eq!(mapped_handle.id, virtual_handle.id);
        assert!(physical_handle.is_mapped());

        // Unmap
        storage.unmap(virtual_handle.id, 0, &mut physical_handle);
        assert!(!physical_handle.is_mapped());

        // Clean up
        storage.free(virtual_handle.id);
        storage.release(physical_handle.id());
    }

    #[test]
    // Demonstrates that we can reserve contiguous address spaces thanks to VMM capabilities.
    fn test_contiguous_address_space_reservations() {
        let mut storage = setup_cuda_storage();
        let granularity = storage.granularity();
        let size = granularity as u64;

        // Reserve two consecutive virtual address spaces
        let handle1 = storage.reserve(size, None).unwrap();
        let handle2 = storage.reserve(size, Some(handle1.id)).unwrap();

        // Test alignment checking (might not be aligned due to CUDA's behavior)
        let are_aligned = storage.are_aligned(&handle1.id, &handle2.id);
        // Note sure if this will always pass (as cuda docs state this is not guaranteed to be always aligned).
        // You can disable it if you desire to.
        // However i have added it to demonstrate that on most cases the hint stuff works (at least is working on my machine).
        // This should allow me to implement an intelligent memory pool that defragments the memory space periodically.
        assert!(are_aligned);

        // Clean up
        storage.free(handle1.id);
        storage.free(handle2.id);
    }

    #[test]
    // Demonstrates that [`reserve`] does not fail even though you allocate more than available memory
    fn test_virtual_memory_overcommitting() {
        let mut storage = setup_cuda_storage();
        let granularity = storage.granularity();
        let size = granularity as u64 * 1000000; // 1000000 * 2 MiB

        let result = storage.reserve(size, None);
        assert!(result.is_ok());

        let handle = result.unwrap();

        // Storage should have stored the handle.
        let space = storage.virtual_memory.get(&handle.id).unwrap();
        assert!(space.ptr() != 0); // Pointer should be valid.
        assert!(space.size() == size); // size should match.
    }
}
