use std::ptr::NonNull;

use cubecl_common::bytes::{Allocation, AllocationController, AllocationError};

pub struct CudaAllocController {
    // Keep the ptr alive for GPU to CPU writes.
    ptr_container: Option<*mut u8>,
}

impl AllocationController for CudaAllocController {
    fn dealloc(&mut self, _allocation: &Allocation) {
        unsafe {
            cudarc::driver::sys::cuMemFreeHost(std::ptr::from_mut(&mut self.ptr_container).cast());
        };
        self.ptr_container = None;
    }

    fn grow(
        &mut self,
        _allocation: &Allocation,
        _size: usize,
        _align: usize,
    ) -> Result<Allocation, AllocationError> {
        Err(AllocationError::UnsupportedOperation)
    }

    fn can_be_detached(&self) -> bool {
        false
    }
}

impl CudaAllocController {
    pub fn init(size: usize, align: usize) -> (Self, Allocation) {
        unsafe {
            let ptr: *mut u8 = std::ptr::null_mut();
            let mut container = Some(ptr);
            let container_ptr = std::ptr::from_mut(&mut container);

            cudarc::driver::sys::cuMemAllocHost_v2(ptr.cast(), size);

            (
                Self {
                    ptr_container: container,
                },
                Allocation {
                    ptr: NonNull::new(ptr).unwrap(),
                    size,
                    align,
                },
            )
        }
    }
}
