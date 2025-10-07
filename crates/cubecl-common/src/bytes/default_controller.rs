//! Module that defines an allocation based on the [alloc crate](alloc).

use crate::bytes::{AllocationController, AllocationError};
use alloc::alloc::Layout;
use alloc::vec::Vec;
use bytemuck::Contiguous;
use core::{alloc::LayoutError, marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

/// The maximum supported alignment. The limit exists to not have to store alignment when serializing. Instead,
/// the bytes are always over-aligned when deserializing to MAX_ALIGN.
pub const MAX_ALIGN: usize = core::mem::align_of::<u128>();

/// Represents a single contiguous memory allocation.
///
/// The allocation can be manipulated using the [AllocationController],
/// though some operations, such as [grow](AllocationController::grow), may not be supported by all
/// implementations.
struct Allocation<'a> {
    /// Points to the beginning of the allocation.
    /// Must be valid for the lifetime of the allocation.
    ptr: NonNull<u8>,
    /// The number of bytes allocated. Nb not all bytes are initialized.
    size: usize,
    /// The memory alignment of the allocation
    align: usize,
    /// The lifetime this allocation is valid for.
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> Allocation<'a> {
    /// Create a new allocation for the given pointer, size, and alignment.
    ///
    /// # Safety
    ///
    /// - Ptr must not be used elsewhere.
    /// - Ptr must be valid for at least the lifetime of `'a`
    /// - The allocation must have the alignment specified by `align`
    /// - The allocation must be at least `size` bytes long
    pub unsafe fn new_init(ptr: NonNull<u8>, size: usize, align: usize) -> Self {
        debug_assert!(
            align <= MAX_ALIGN,
            "alignment exceeds maximum supported alignment"
        );
        debug_assert!(
            ptr.as_ptr().align_offset(align.into_integer()) == 0,
            "pointer is not properly aligned"
        );

        Self {
            ptr,
            size,
            align,
            _lifetime: PhantomData,
        }
    }

    fn dangling(align: usize) -> Allocation<'a> {
        let ptr = core::ptr::null_mut::<u8>().wrapping_add(align);
        Self {
            ptr: NonNull::new(ptr).unwrap(),
            size: 0,
            align,
            _lifetime: PhantomData,
        }
    }
}

/// Allocation controller using the 'native' rust [alloc crate](alloc).
///
/// SAFETY:
///  - If `layout.size() > 0`, `ptr` points to a valid allocation from the global allocator
///    of the specified layout. The first `len` bytes are initialized.
///  - If `layout.size() == 0`, `ptr` is aligned to `layout.align()` and `len` is 0.
///    `ptr` is further suitable to be used as the argument for `Vec::from_raw_parts` see [buffer alloc]
///    for more details.
pub(crate) struct NativeAllocationController<'a> {
    allocation: Allocation<'a>,
}

impl<'a> NativeAllocationController<'a> {
    pub(crate) fn alloc_with_data(data: &[u8], align: usize) -> Result<Self, LayoutError> {
        debug_assert!(
            align <= MAX_ALIGN,
            "alignment exceeds maximum supported alignment"
        );

        // Round up capacity to next multiple of alignment
        let capacity = data.len().next_multiple_of(align.into_integer());
        let mut controller = Self::alloc_with_capacity(capacity, align)?;
        // See also #![feature(maybe_uninit_write_slice)], which would replace this with safe code
        // SAFETY: reinterpreting the slice as a MaybeUninit<u8>, and only reading from it.
        unsafe {
            controller.memory_mut()[..data.len()].copy_from_slice(core::slice::from_raw_parts(
                data.as_ptr().cast(),
                data.len(),
            ));
        }
        Ok(controller)
    }

    pub(crate) fn alloc_with_capacity(capacity: usize, align: usize) -> Result<Self, LayoutError> {
        debug_assert!(
            align <= MAX_ALIGN,
            "alignment exceeds maximum supported alignment"
        );
        debug_assert!(
            capacity.is_multiple_of(align.into_integer()),
            "capacity must be a multiple of alignment"
        );

        let layout = Layout::from_size_align(capacity, align.into_integer())?;
        let ptr = buffer_alloc(layout);

        // SAFETY:
        // - The pointer is valid until the controller is deallocated.
        // - The pointer was allocated with the given layout.
        let allocation = unsafe { Allocation::new_init(ptr, layout.size(), layout.align()) };

        // The allocation was done with the core allocator.
        Ok(Self { allocation })
    }

    pub(crate) fn from_elems<E>(elems: Vec<E>) -> Self
    where
        E: bytemuck::NoUninit + Send + Sync,
    {
        let mut elems = core::mem::ManuallyDrop::new(elems);

        // Set the length to 0, then all data is in the "spare capacity".
        // The data is Copy, so in particular does not need to be dropped. In any case, try not to panic until
        // we have taken ownership of the data!
        //
        // SAFETY:
        // - 0 is less or equal to capacity.
        // - 0 elements are all initialized elements.
        unsafe { elems.set_len(0) };

        // We now have one contiguous slice of data to pass to Layout::for_value.
        let data = elems.spare_capacity_mut();
        let layout = Layout::for_value(data);
        let ptr = NonNull::new(elems.as_mut_ptr() as *mut u8).unwrap();

        // SAFETY:
        // - The pointer is valid as long as the controller is alive and not dealloced.
        // - The allocation was done with the size and layout of the data.
        let alloc = unsafe { Allocation::new_init(ptr, layout.size(), layout.align()) };

        Self { allocation: alloc }
    }
}

impl AllocationController for NativeAllocationController<'_> {
    fn grow(&mut self, size: usize, align: usize) -> Result<(), AllocationError> {
        debug_assert!(
            align <= MAX_ALIGN,
            "alignment exceeds maximum supported alignment"
        );
        debug_assert!(
            size > self.allocation.size,
            "new size must be larger than current size"
        );

        let Ok(new_layout) = Layout::from_size_align(size, align) else {
            return Err(AllocationError::OutOfMemory);
        };

        // SAFETY: Check done before.
        let old_layout = unsafe {
            Layout::from_size_align_unchecked(self.allocation.size, self.allocation.align)
        };
        let (layout, ptr) = buffer_grow(old_layout, self.allocation.ptr, new_layout);

        // SAFETY:
        // - The ptr is valid for 'a lifetime as we own the binding for 'a.
        // - We allocated with the size and align of the layout.
        self.allocation = unsafe { Allocation::new_init(ptr, layout.size(), layout.align()) };

        Ok(())
    }

    // SAFETY: Per type invariants, we only take in memory allocated with the rust core allocator.
    fn try_detach(&mut self) -> Option<NonNull<u8>> {
        let ptr = self.allocation.ptr;
        self.allocation = Allocation::dangling(self.allocation.align);
        Some(ptr)
    }

    fn alloc_align(&self) -> usize {
        self.allocation.align
    }

    fn memory(&self) -> &[MaybeUninit<u8>] {
        unsafe {
            core::slice::from_raw_parts(self.allocation.ptr.as_ptr().cast(), self.allocation.size)
        }
    }

    /// # Safety
    /// - Only initialized memory must be written to this slice.
    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY:
        // - Ptr is valid per type invariants.
        // - Size is valid per type invariants.
        // - Caller promises that only initialized memory is written to this slice.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.allocation.ptr.as_ptr().cast(),
                self.allocation.size,
            )
        }
    }
}

impl Drop for NativeAllocationController<'_> {
    fn drop(&mut self) {
        let layout = unsafe {
            Layout::from_size_align_unchecked(self.allocation.size, self.allocation.align)
        };
        buffer_dealloc(layout, self.allocation.ptr.cast());
    }
}

// Allocate a pointer that can be passed to Vec::from_raw_parts
fn buffer_alloc(layout: Layout) -> NonNull<u8> {
    // [buffer alloc]: The current docs of Vec::from_raw_parts(ptr, ...) say:
    //   > ptr must have been allocated using the global allocator
    // Yet, an empty Vec is guaranteed to not allocate (it is even illegal! to allocate with a zero-sized layout)
    // Hence, we slightly re-interpret the above to only needing to hold if `capacity > 0`. Still, the pointer
    // must be non-zero. So in case we need a pointer for an empty vec, use a correctly aligned, dangling one.
    if layout.size() == 0 {
        // we would use NonNull:dangling() but we don't have a concrete type for the requested alignment
        let ptr = core::ptr::null_mut::<u8>().wrapping_add(layout.align());
        NonNull::new(ptr).unwrap()
    } else {
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc::alloc(layout) };
        NonNull::new(ptr.cast()).unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout))
    }
}

// Deallocate a buffer of a Vec
fn buffer_dealloc(layout: Layout, buffer: NonNull<u8>) {
    if layout.size() != 0 {
        // SAFETY: buffer comes from a Vec or from [`buffer_alloc`/`buffer_grow`].
        // The layout is the same as per type-invariants
        unsafe {
            alloc::alloc::dealloc(buffer.as_ptr().cast(), layout);
        }
    } else {
        // An empty Vec does not allocate, hence nothing to dealloc
        expect_dangling(layout.align(), buffer.cast());
    }
}

// Grow the buffer while keeping alignment
fn buffer_grow(
    old_layout: Layout,
    buffer: NonNull<u8>,
    min_layout: Layout,
) -> (Layout, NonNull<u8>) {
    let new_align = min_layout.align().max(old_layout.align()); // Don't let data become less aligned
    let new_size = min_layout.size().next_multiple_of(new_align);
    if new_size > isize::MAX as usize {
        alloc_overflow();
    }

    assert!(new_size > old_layout.size(), "size must actually grow");
    if old_layout.size() == 0 {
        expect_dangling(old_layout.align(), buffer);
        let new_layout = Layout::from_size_align(new_size, new_align).unwrap();
        let buffer = buffer_alloc(new_layout);
        return (new_layout, buffer);
    };
    let realloc = || {
        let new_layout = Layout::from_size_align(new_size, old_layout.align()).unwrap();
        // SAFETY:
        // - buffer comes from a Vec or from [`buffer_alloc`/`buffer_grow`].
        // - old_layout is the same as with which the pointer was allocated
        // - new_size is not 0, since it is larger than old_layout.size() which is non-zero
        // - size constitutes a valid layout
        let ptr = unsafe { alloc::alloc::realloc(buffer.as_ptr(), old_layout, new_layout.size()) };
        (new_layout, ptr)
    };
    if new_align == old_layout.align() {
        // happy path. We can just realloc.
        let (new_layout, ptr) = realloc();
        let buffer = NonNull::new(ptr);
        let buffer = buffer.unwrap_or_else(|| alloc::alloc::handle_alloc_error(new_layout));
        return (new_layout, buffer);
    }
    // [buffer grow]: alloc::realloc can *not* change the alignment of the allocation's layout.
    // The unstable Allocator::{grow,shrink} API changes this, but might take a while to make it
    // into alloc::GlobalAlloc.
    //
    // As such, we can not request a specific alignment. But most allocators will give us the required
    // alignment "for free". Hence, we speculatively avoid a mem-copy by using realloc.
    //
    // If in the future requesting an alignment change for an existing is available, this can be removed.
    #[cfg(target_has_atomic = "8")]
    mod alignment_assumption {
        use core::sync::atomic::{AtomicBool, Ordering};
        static SPECULATE: AtomicBool = AtomicBool::new(true);
        pub fn speculate() -> bool {
            // We load and store with relaxed order, since worst case this leads to a few more memcopies
            SPECULATE.load(Ordering::Relaxed)
        }
        pub fn report_violation() {
            SPECULATE.store(false, Ordering::Relaxed)
        }
    }
    #[cfg(not(target_has_atomic = "8"))]
    mod alignment_assumption {
        // On these platforms we don't speculate, and take the hit of performance
        pub fn speculate() -> bool {
            false
        }
        pub fn report_violation() {}
    }
    // reminder: old_layout.align() < new_align
    let mut old_buffer = buffer;
    let mut old_layout = old_layout;
    if alignment_assumption::speculate() {
        let (realloc_layout, ptr) = realloc();
        if let Some(buffer) = NonNull::new(ptr) {
            if buffer.align_offset(new_align) == 0 {
                return (realloc_layout, buffer);
            }
            // Speculating hasn't succeeded, but access now has to go through the reallocated buffer
            alignment_assumption::report_violation();
            old_buffer = buffer.cast();
            old_layout = realloc_layout;
        } else {
            // If realloc fails, the later alloc will likely too, but don't report this yet
        }
    }
    // realloc but change alignment. This requires a mem copy as pointed out above
    let new_layout = Layout::from_size_align(new_size, new_align).unwrap();
    let new_buffer = buffer_alloc(new_layout);
    // SAFETY: two different memory allocations, and old buffer's size is smaller than new_size
    unsafe {
        core::ptr::copy_nonoverlapping(
            old_buffer.as_ptr(),
            new_buffer.as_ptr().cast(),
            old_layout.size(),
        );
    }
    buffer_dealloc(old_layout, old_buffer);
    (new_layout, new_buffer)
}

fn expect_dangling(align: usize, buffer: NonNull<u8>) {
    debug_assert!(
        buffer.as_ptr().wrapping_sub(align).is_null(),
        "expected a nullptr for size 0"
    );
}

#[cold]
pub fn alloc_overflow() -> ! {
    panic!("Overflow, too many elements")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::AllocationController;

    #[test]
    fn test_core_allocation_controller_alloc_with_capacity() {
        let controller = NativeAllocationController::alloc_with_capacity(64, 8).unwrap();
        assert_eq!(controller.alloc_align(), 8);
        assert_eq!(controller.memory().len(), 64);
    }

    #[test]
    fn test_core_allocation_controller_alloc_with_data() {
        let data = b"hello world test"; // 16 bytes to be multiple of 8
        let controller = NativeAllocationController::alloc_with_data(data, 8).unwrap();
        assert_eq!(controller.alloc_align(), 8);
        assert!(controller.memory().len() >= data.len());
        assert_eq!(controller.memory().len() % 8, 0); // Should be multiple of alignment

        // Verify data was copied correctly
        let memory = controller.memory();
        let memory_slice =
            unsafe { core::slice::from_raw_parts(memory.as_ptr() as *const u8, data.len()) };
        assert_eq!(memory_slice, data);
    }

    #[test]
    fn test_core_allocation_controller_from_elems() {
        let elems = vec![1u32, 2, 3, 4];
        let expected_bytes = elems.len() * core::mem::size_of::<u32>();

        let controller = NativeAllocationController::from_elems(elems);
        assert_eq!(controller.alloc_align(), core::mem::align_of::<u32>());
        assert_eq!(controller.memory().len(), expected_bytes);
    }

    #[test]
    fn test_core_allocation_controller_grow() {
        let mut controller = NativeAllocationController::alloc_with_capacity(32, 8).unwrap();
        let old_memory_len = controller.memory().len();

        controller.grow(64, 8).unwrap();

        assert_eq!(controller.alloc_align(), 8);
        assert!(controller.memory().len() >= 64);
        assert!(controller.memory().len() > old_memory_len);
    }

    #[test]
    fn test_buffer_alloc_zero_size() {
        let layout = Layout::from_size_align(0, 8).unwrap();
        let ptr = buffer_alloc(layout);
        assert_eq!(ptr.as_ptr().align_offset(8), 0);
        buffer_dealloc(layout, ptr);
    }

    #[test]
    fn test_buffer_grow_from_zero() {
        let old_layout = Layout::from_size_align(0, 8).unwrap();
        let buffer = buffer_alloc(old_layout);
        let min_layout = Layout::from_size_align(64, 8).unwrap();
        let (new_layout, new_buffer) = buffer_grow(old_layout, buffer, min_layout);
        assert!(new_layout.size() >= 64);
        assert_eq!(new_layout.align(), 8);
        buffer_dealloc(new_layout, new_buffer);
    }

    #[test]
    fn test_memory_access() {
        let data = b"test data"; // 9 bytes, will be rounded up to 16 for 8-byte alignment
        let controller = NativeAllocationController::alloc_with_data(data, 8).unwrap();

        let memory = controller.memory();
        assert!(memory.len() >= data.len());
        assert_eq!(memory.len() % 8, 0); // Should be multiple of alignment
        let memory_slice =
            unsafe { core::slice::from_raw_parts(memory.as_ptr() as *const u8, data.len()) };
        assert_eq!(memory_slice, data);
    }

    #[test]
    fn test_memory_mut_access() {
        let mut controller = NativeAllocationController::alloc_with_capacity(16, 8).unwrap();
        unsafe {
            let memory = controller.memory_mut();
            assert_eq!(memory.len(), 16);
            // Write some test data
            memory[0].write(42);
            memory[1].write(84);
        }
        // Verify the data was written
        let memory = controller.memory();
        unsafe {
            assert_eq!(memory[0].assume_init(), 42);
            assert_eq!(memory[1].assume_init(), 84);
        }
    }

    #[test]
    #[should_panic(expected = "capacity must be a multiple of alignment")]
    fn test_debug_assert_capacity_alignment_mismatch() {
        let _ = NativeAllocationController::alloc_with_capacity(33, 8);
    }
}
