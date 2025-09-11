use crate::bytes::default_controller::{self, DefaultAllocationController};
use crate::bytes::{Allocation, AllocationController};
use core::alloc::Layout;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

/// Internal representation of a buffer.
///
/// A buffer is composed of an [allocation](Allocation) and a dynamic [controller](AllocationController).
pub struct Buffer {
    pub allocation: Allocation,
    pub controller: Box<dyn AllocationController>,
}

impl Buffer {
    // Wrap the allocation of a vector without copying
    pub(crate) fn from_vec<E: Copy>(vec: Vec<E>) -> Self {
        let mut elems = core::mem::ManuallyDrop::new(vec);
        // Set the length to 0, then all data is in the "spare capacity".
        // SAFETY: Data is Copy, so in particular does not need to be dropped. In any case, try not to panic until
        //  we have taken ownership of the data!
        unsafe { elems.set_len(0) };
        let data = elems.spare_capacity_mut();
        // We now have one contiguous slice of data to pass to Layout::for_value.
        let layout = Layout::for_value(data);
        // SAFETY: data is the allocation of a vec, hence can not be null. We use unchecked to avoid a panic-path.
        let ptr = unsafe { NonNull::new_unchecked(elems.as_mut_ptr().cast()) };
        let controller = DefaultAllocationController;
        let allocation = Allocation {
            ptr,
            size: layout.size(),
            align: layout.align(),
        };

        Self {
            allocation,
            controller: Box::new(controller),
        }
    }

    // Create a new allocation with the specified layout
    pub fn new(layout: Layout) -> Self {
        let ptr = default_controller::buffer_alloc(layout);
        let allocation = Allocation {
            ptr,
            size: layout.size(),
            align: layout.align(),
        };
        let contoller = DefaultAllocationController;
        Self {
            allocation,
            controller: Box::new(contoller),
        }
    }

    // Returns a mutable view of the memory of the whole allocation
    pub fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: See type invariants
        unsafe {
            core::slice::from_raw_parts_mut(
                self.allocation.ptr.as_ptr().cast(),
                self.allocation.size,
            )
        }
    }

    pub fn grow(&mut self, size: usize, align: usize) {
        match self.controller.grow(&self.allocation, size, align) {
            Ok(allocation) => {
                self.allocation = allocation;
            }
            Err(_err) => {
                let mut new = Self::new(Layout::from_size_align(size, align).unwrap());
                let data = unsafe {
                    core::slice::from_raw_parts(
                        self.allocation.ptr.as_ptr().cast(),
                        self.allocation.size,
                    )
                };
                new.memory_mut().copy_from_slice(data);
                *self = new;
            }
        }
    }

    // Return a pointer to the underlying allocation. This pointer is valid for reads and writes until the allocation is dropped or reallocated.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.allocation.ptr.as_ptr()
    }

    // Try to convert the allocation to a Vec. The Vec has a length of 0 when returned, but correct capacity and pointer!
    pub fn try_into_vec<E>(self) -> Result<Vec<E>, Self> {
        if !self.controller.can_be_detached() {
            return Err(self);
        }

        let byte_capacity = self.allocation.size;
        let Some(capacity) = byte_capacity.checked_div(size_of::<E>()) else {
            return Err(self);
        };
        if capacity * size_of::<E>() != byte_capacity {
            return Err(self);
        };
        if self.allocation.align < align_of::<E>() {
            return Err(self);
        }
        // Okay, let's commit
        let ptr = self.allocation.ptr.as_ptr().cast();
        core::mem::forget(self);
        // SAFETY:
        // - ptr was allocated by the global allocator as per type-invariant
        // - `E` has the same alignment as indicated by the stored layout.
        // - capacity * size_of::<E> == layout.size()
        // - 0 <= capacity
        // - no bytes are claimed to be initialized
        // - the layout represents a valid allocation, hence has allocation size less than isize::MAX
        Ok(unsafe { Vec::from_raw_parts(ptr, 0, capacity) })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.controller.dealloc(&self.allocation);
    }
}
