use std::{ffi::c_void, ptr::NonNull};

use cubecl_common::bytes::{Allocation, AllocationController, AllocationError};

pub struct CudaAllocController {
    // Keep the ptr alive for GPU to CPU writes.
    ptr2ptr: *mut *mut c_void,
}

impl AllocationController for CudaAllocController {
    fn dealloc(&mut self, allocation: &Allocation) {
        unsafe {
            // cudarc::driver::sys::cuMemFreeHost(allocation.ptr.cast());
        };
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
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let ptr2ptr: *mut *mut c_void = &mut ptr;

            // Call cuMemAllocHost_v2 to allocate pinned host memory
            let result = cudarc::driver::sys::cuMemAllocHost_v2(ptr2ptr, size);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!("cuMemAllocHost_v2 failed with error code: {:?}", result);
            }

            // Ensure ptr is not null
            if ptr.is_null() {
                panic!("cuMemAllocHost_v2 returned a null pointer");
            }

            (
                Self { ptr2ptr },
                Allocation {
                    ptr: NonNull::new(ptr as *mut u8).expect("NonNull creation failed"),
                    size,
                    align,
                },
            )
        }
    }
}
