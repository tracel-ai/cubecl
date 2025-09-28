//! Module that defines an allocation based on the [alloc crate](alloc).

use crate::bytes::{AllocationController, AllocationError};
use alloc::alloc::Layout;
use core::{alloc::LayoutError, marker::PhantomData, mem::MaybeUninit, num::NonZero, ptr::NonNull};

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
        Self {
            ptr,
            size,
            align,
            _lifetime: PhantomData,
        }
    }
}

/// Allocation controller using the core rust [alloc crate](alloc).
///
/// SAFETY:
///  - If `layout.size() > 0`, `ptr` points to a valid allocation from the global allocator
///    of the specified layout. The first `len` bytes are initialized.
///  - If `layout.size() == 0`, `ptr` is aligned to `layout.align()` and `len` is 0.
///    `ptr` is further suitable to be used as the argument for `Vec::from_raw_parts` see [buffer alloc]
///    for more details.
pub(crate) struct CoreAllocationController<'a> {
    allocation: Allocation<'a>,
}

impl<'a> CoreAllocationController<'a> {
    pub(crate) fn alloc_with_data(data: &[u8], align: usize) -> Result<Self, LayoutError> {
        let mut controller = Self::alloc_with_capacity(data.len(), align)?;
        // See also #![feature(maybe_uninit_write_slice)], which would replace this with safe code
        // SAFETY: reinterpreting the slice as a MaybeUninit<u8>, and only reading from it.
        unsafe {
            controller
                .memory_mut()
                .copy_from_slice(core::slice::from_raw_parts(
                    data.as_ptr().cast(),
                    data.len(),
                ));
        }
        Ok(controller)
    }

    pub(crate) fn alloc_with_capacity(capacity: usize, align: usize) -> Result<Self, LayoutError> {
        assert!(capacity.is_multiple_of(align));
        let layout = Layout::from_size_align(capacity, align)?;
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

impl Drop for CoreAllocationController<'_> {
    fn drop(&mut self) {
        let layout = unsafe {
            Layout::from_size_align_unchecked(self.allocation.size, self.allocation.align)
        };
        buffer_dealloc(layout, self.allocation.ptr.cast());
    }
}

impl AllocationController for CoreAllocationController<'_> {
    fn grow(&mut self, size: usize, align: usize) -> Result<(), AllocationError> {
        let Ok(new_layout) = Layout::from_size_align(size, align) else {
            return Err(AllocationError::OutOfMemory);
        };

        // SAFETY: Check done before.
        let old_layout = unsafe {
            Layout::from_size_align_unchecked(self.allocation.size, self.allocation.align)
        };
        let (layout, ptr) = buffer_grow(old_layout, self.allocation.ptr.cast(), new_layout);

        // SAFETY:
        // - The ptr is valid for 'a lifetime as we own the binding for 'a.
        // - We allocated with the size and align of the layout.
        self.allocation = unsafe { Allocation::new_init(ptr, layout.size(), layout.align()) };

        Ok(())
    }

    // SAFETY: Per type invariants, we only take in memory allocated with the rust core allocator.
    fn can_be_detached(&self) -> bool {
        true
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

// Allocate a pointer that can be passed to Vec::from_raw_parts
pub(crate) fn buffer_alloc(layout: Layout) -> NonNull<u8> {
    // [buffer alloc]: The current docs of Vec::from_raw_parts(ptr, ...) say:
    //   > ptr must have been allocated using the global allocator
    // Yet, an empty Vec is guaranteed to not allocate (it is even illegal! to allocate with a zero-sized layout)
    // Hence, we slightly re-interpret the above to only needing to hold if `capacity > 0`. Still, the pointer
    // must be non-zero. So in case we need a pointer for an empty vec, use a correctly aligned, dangling one.
    if layout.size() == 0 {
        // we would use NonNull:dangling() but we don't have a concrete type for the requested alignment
        // SAFETY: layout.align() is never 0
        NonNull::without_provenance(unsafe { NonZero::new_unchecked(layout.align()) })
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
