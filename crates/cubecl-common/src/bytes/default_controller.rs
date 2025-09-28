//! Module that defines an allocation based on the [alloc crate](alloc).

use crate::bytes::{AllocationController, AllocationError, allocation::Allocation};
use alloc::alloc::Layout;
use core::{mem::MaybeUninit, num::NonZero, ptr::NonNull};

/// The maximum supported alignment. The limit exists to not have to store alignment when serializing. Instead,
/// the bytes are always over-aligned when deserializing to MAX_ALIGN.
pub const MAX_ALIGN: usize = core::mem::align_of::<u128>();

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
    // SAFETY:
    // - Allocation must be allocated with the rust core allocator.
    pub(crate) unsafe fn new(allocation: Allocation<'a>) -> Self {
        Self { allocation }
    }
}

impl Drop for CoreAllocationController<'_> {
    fn drop(&mut self) {
        let layout = unsafe {
            Layout::from_size_align_unchecked(
                self.allocation.memory().len(),
                self.allocation.align(),
            )
        };
        buffer_dealloc(layout, self.allocation.ptr());
    }
}

impl AllocationController for CoreAllocationController<'_> {
    fn grow(&mut self, size: usize, align: usize) -> Result<(), AllocationError> {
        let Ok(new_layout) = Layout::from_size_align(size, align) else {
            return Err(AllocationError::OutOfMemory);
        };

        // Check done before.
        let old_layout = unsafe {
            Layout::from_size_align_unchecked(
                self.allocation.memory().len(),
                self.allocation.align(),
            )
        };
        let (layout, ptr) = buffer_grow(old_layout, self.allocation.ptr(), new_layout);

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
        self.allocation.align()
    }

    fn memory(&self) -> &[MaybeUninit<u8>] {
        self.allocation.memory()
    }

    fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.allocation.memory_mut()
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
fn buffer_dealloc(layout: Layout, buffer: NonNull<MaybeUninit<u8>>) {
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
    buffer: NonNull<MaybeUninit<u8>>,
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
        let ptr =
            unsafe { alloc::alloc::realloc(buffer.as_ptr().cast(), old_layout, new_layout.size()) };
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

fn expect_dangling(align: usize, buffer: NonNull<MaybeUninit<u8>>) {
    debug_assert!(
        buffer.as_ptr().wrapping_sub(align).is_null(),
        "expected a nullptr for size 0"
    );
}

#[cold]
pub fn alloc_overflow() -> ! {
    panic!("Overflow, too many elements")
}
