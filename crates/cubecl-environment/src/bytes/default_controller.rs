//! Module that defines an allocation controller over a [`BufferAllocator`], and the default
//! provider that uses the [alloc crate](alloc).

use crate::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationError, BufferAllocator,
};
use alloc::alloc::Layout;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::{mem::MaybeUninit, num::NonZeroUsize, ptr::NonNull};

use super::AllocationProperty;

/// The maximum supported alignment. The limit exists to not have to store alignment when serializing. Instead,
/// the bytes are always over-aligned when deserializing to `MAX_ALIGN`.
pub const MAX_ALIGN: usize = core::mem::align_of::<u128>();

/// Where the memory of a [`Controller`] comes from.
///
/// Only this crate implements the trait.
pub(crate) trait Provider: BufferAllocator + Copy + 'static {
    /// Whether the memory may become a `Vec`. Only memory from the global allocator may,
    /// because a `Vec` releases its memory with the global allocator.
    const DETACHABLE: bool;
    /// The property that the controller reports.
    const PROPERTY: AllocationProperty;
}

/// The global allocator of the [alloc crate](alloc).
///
/// Like `Vec`, it aborts through [`handle_alloc_error`](alloc::alloc::handle_alloc_error) when
/// the global allocator has no memory, and never returns an error.
#[derive(Clone, Copy)]
pub(crate) struct Global;

impl Provider for Global {
    const DETACHABLE: bool = true;
    const PROPERTY: AllocationProperty = AllocationProperty::Native;
}

// SAFETY: The global allocator gives the guarantees of `BufferAllocator`.
unsafe impl BufferAllocator for Global {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocationError> {
        // SAFETY: `BufferAllocator::allocate` never receives a layout of zero size.
        let ptr = unsafe { alloc::alloc::alloc(layout) };
        Ok(NonNull::new(ptr).unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout)))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: `ptr` comes from a Vec or from `allocate`/`grow`, with this layout.
        unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<u8>, AllocationError> {
        let realloc = |ptr: NonNull<u8>, old: Layout| {
            // SAFETY:
            // - ptr comes from a Vec or from `allocate`/`grow`, with layout `old`.
            // - new.size() is not 0, since it is at least old.size() which is non-zero.
            let ptr = unsafe { alloc::alloc::realloc(ptr.as_ptr(), old, new.size()) };
            // SAFETY: The size comes from a valid layout, and the alignment from another.
            let layout = unsafe { Layout::from_size_align_unchecked(new.size(), old.align()) };
            (layout, NonNull::new(ptr))
        };
        if new.align() == old.align() {
            // happy path. We can just realloc.
            let (layout, ptr) = realloc(ptr, old);
            return Ok(ptr.unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout)));
        }
        // [buffer grow]: alloc::realloc can *not* change the alignment of the allocation's layout.
        // The unstable Allocator::{grow,shrink} API changes this, but might take a while to make it
        // into alloc::GlobalAlloc.
        //
        // As such, we can not request a specific alignment. But most allocators will give us the required
        // alignment "for free". Hence, we speculatively avoid a mem-copy by using realloc.
        //
        // If in the future requesting an alignment change for an existing is available, this can be removed.
        let mut old_buffer = ptr;
        let mut old_layout = old;
        if alignment_assumption::speculate() {
            // If realloc fails, the later alloc will likely too, but don't report this yet
            if let (layout, Some(buffer)) = realloc(ptr, old) {
                if buffer.align_offset(new.align()) == 0 {
                    return Ok(buffer);
                }
                // Speculating hasn't succeeded, but access now has to go through the reallocated buffer
                alignment_assumption::report_violation();
                old_buffer = buffer;
                old_layout = layout;
            }
        }
        // realloc but change alignment. This requires a mem copy as pointed out above
        let new_buffer = self.allocate(new)?;
        // SAFETY: two different memory allocations, and old buffer's size is not larger than new.size()
        unsafe {
            core::ptr::copy_nonoverlapping(
                old_buffer.as_ptr(),
                new_buffer.as_ptr(),
                old_layout.size(),
            );
            self.deallocate(old_buffer, old_layout);
        }
        Ok(new_buffer)
    }
}

/// Whether the global allocator returns more alignment than it is asked for, on a `realloc`.
/// Only [`Global`] uses it, so a custom allocator never changes it.
#[cfg(target_has_atomic = "8")]
mod alignment_assumption {
    use crate::sync::{AtomicBool, Ordering};
    static SPECULATE: AtomicBool = AtomicBool::new(true);
    pub fn speculate() -> bool {
        // We load and store with relaxed order, since worst case this leads to a few more memcopies
        SPECULATE.load(Ordering::Relaxed)
    }
    pub fn report_violation() {
        SPECULATE.store(false, Ordering::Relaxed);
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

/// Allocation controller whose memory comes from a [`Provider`].
///
/// SAFETY:
///  - If `size > 0`, `ptr` points to a valid allocation of `provider` with layout `size` and
///    `align`. Not all bytes are initialized.
///  - If `size == 0`, `ptr` is aligned to `align` and dangling, and no memory is allocated.
///    `ptr` is further suitable to be used as the argument for `Vec::from_raw_parts`, see
///    [`Controller::from_elems`].
///  - [`try_detach`](AllocationController::try_detach) returns the memory only when
///    [`Provider::DETACHABLE`] is `true`.
pub(crate) struct Controller<P: Provider> {
    provider: P,
    ptr: NonNull<u8>,
    size: usize,
    align: usize,
}

/// Allocation controller using the 'native' rust [alloc crate](alloc).
pub(crate) type NativeAllocationController = Controller<Global>;

impl<P: Provider> Controller<P> {
    /// Allocates `capacity` uninitialized bytes from `provider`.
    ///
    /// `capacity` must be a multiple of `align`.
    pub(crate) fn alloc_with_capacity(
        provider: P,
        capacity: usize,
        align: usize,
    ) -> Result<Self, AllocationError> {
        debug_assert!(
            capacity.is_multiple_of(align),
            "capacity must be a multiple of alignment"
        );

        let layout = layout(capacity, align)?;
        let ptr = if capacity == 0 {
            dangling(align)
        } else {
            provider.allocate(layout)?
        };
        debug_assert!(
            ptr.as_ptr().align_offset(align) == 0,
            "pointer is not properly aligned"
        );

        Ok(Self {
            provider,
            ptr,
            size: capacity,
            align,
        })
    }

    /// Copies `data` into new memory from `provider`, aligned to `align`.
    pub(crate) fn alloc_with_data(
        provider: P,
        data: &[u8],
        align: usize,
    ) -> Result<Self, AllocationError> {
        // Round up capacity to next multiple of alignment
        let capacity = data.len().next_multiple_of(align);
        let mut controller = Self::alloc_with_capacity(provider, capacity, align)?;
        // See also #![feature(maybe_uninit_write_slice)], which would replace this with safe code
        // SAFETY: reinterpreting the slice as a MaybeUninit<u8>, and only reading from it.
        unsafe {
            controller
                .memory_mut(AccessPolicy::default())
                .expect("controller memory is always host-resident")[..data.len()]
                .copy_from_slice(core::slice::from_raw_parts(
                    data.as_ptr().cast(),
                    data.len(),
                ));
        }
        Ok(controller)
    }

    fn layout(&self) -> Layout {
        // SAFETY: The size and the alignment come from a valid layout.
        unsafe { Layout::from_size_align_unchecked(self.size, self.align) }
    }
}

impl NativeAllocationController {
    /// Adopts the buffer of `elems`, without a copy.
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
        // An empty Vec does not allocate, and its dangling pointer is aligned to `E`: that is the
        // same pointer that `dangling` makes.
        let ptr = NonNull::new(elems.as_mut_ptr().cast::<u8>()).unwrap();

        // The pointer is valid as long as the controller is alive and not dealloced, and the
        // allocation was done by the global allocator with the size and layout of the data.
        Self {
            provider: Global,
            ptr,
            size: layout.size(),
            align: layout.align(),
        }
    }
}

/// Makes the layout of an allocation, with the alignment limit of [`Bytes`](super::Bytes).
fn layout(size: usize, align: usize) -> Result<Layout, AllocationError> {
    if align > MAX_ALIGN {
        return Err(AllocationError::UnsupportedOperation);
    }
    Layout::from_size_align(size, align).map_err(|_| AllocationError::OutOfMemory)
}

/// A pointer for an allocation of zero bytes.
fn dangling(align: usize) -> NonNull<u8> {
    // [buffer alloc]: The current docs of Vec::from_raw_parts(ptr, ...) say:
    //   > ptr must have been allocated using the global allocator
    // Yet, an empty Vec is guaranteed to not allocate (it is even illegal! to allocate with a zero-sized layout)
    // Hence, we slightly re-interpret the above to only needing to hold if `capacity > 0`. Still, the pointer
    // must be non-zero. So in case we need a pointer for an empty vec, use a correctly aligned, dangling one.
    NonNull::without_provenance(NonZeroUsize::new(align).expect("alignment is never zero"))
}

impl<P: Provider> AllocationController for Controller<P> {
    fn grow(&mut self, size: usize, align: usize) -> Result<(), AllocationError> {
        debug_assert!(
            align <= MAX_ALIGN,
            "alignment exceeds maximum supported alignment"
        );
        debug_assert!(
            size > self.size,
            "new size must be larger than current size"
        );

        // Don't let data become less aligned
        let align = align.max(self.align);
        let new = layout(size.next_multiple_of(align), align)?;
        let ptr = if self.size == 0 {
            self.provider.allocate(new)?
        } else {
            // SAFETY: Per type invariants, `ptr` comes from `provider` with this layout, and
            // the new size is larger.
            unsafe { self.provider.grow(self.ptr, self.layout(), new)? }
        };

        self.ptr = ptr;
        self.size = new.size();
        self.align = new.align();
        Ok(())
    }

    fn property(&self) -> AllocationProperty {
        P::PROPERTY
    }

    // SAFETY: Per type invariants, only memory from the global allocator is detachable.
    fn try_detach(&mut self) -> Option<NonNull<u8>> {
        if !P::DETACHABLE {
            return None;
        }
        let ptr = self.ptr;
        self.ptr = dangling(self.align);
        self.size = 0;
        Some(ptr)
    }

    fn alloc_align(&self) -> usize {
        self.align
    }

    // The memory is always host-resident: the policy never forces a copy here.
    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        // SAFETY: Per type invariants, `ptr` is valid for `size` bytes.
        Ok(unsafe { core::slice::from_raw_parts(self.ptr.as_ptr().cast(), self.size) })
    }

    /// # Safety
    /// - Only initialized memory must be written to this slice.
    unsafe fn memory_mut(
        &mut self,
        _policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        // SAFETY:
        // - Ptr is valid per type invariants.
        // - Size is valid per type invariants.
        // - Caller promises that only initialized memory is written to this slice.
        Ok(unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.size) })
    }

    // Copies stay in the memory of `provider`.
    fn alloc_like(
        &self,
        capacity: usize,
        align: usize,
    ) -> Option<Result<Box<dyn AllocationController>, AllocationError>> {
        Some(
            Self::alloc_with_capacity(self.provider, capacity, align)
                .map(|controller| Box::new(controller) as Box<dyn AllocationController>),
        )
    }
}

impl<P: Provider> Drop for Controller<P> {
    fn drop(&mut self) {
        if self.size != 0 {
            // SAFETY: Per type invariants, `ptr` comes from `provider` with this layout.
            unsafe { self.provider.deallocate(self.ptr, self.layout()) }
        }
    }
}

#[cold]
pub fn alloc_overflow() -> ! {
    panic!("Overflow, too many elements")
}

// The type is no-std; its tests need std (test_log, std collections).
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::bytes::{AccessPolicy, AllocationController};
    use alloc::vec;

    #[test_log::test]
    fn test_core_allocation_controller_alloc_with_capacity() {
        let controller = NativeAllocationController::alloc_with_capacity(Global, 64, 8).unwrap();
        assert_eq!(controller.alloc_align(), 8);
        assert_eq!(
            controller.memory(AccessPolicy::default()).unwrap().len(),
            64
        );
    }

    #[test_log::test]
    fn test_core_allocation_controller_alloc_with_data() {
        let data = b"hello world test"; // 16 bytes to be multiple of 8
        let controller = NativeAllocationController::alloc_with_data(Global, data, 8).unwrap();
        assert_eq!(controller.alloc_align(), 8);
        assert!(controller.memory(AccessPolicy::default()).unwrap().len() >= data.len());
        assert_eq!(
            controller.memory(AccessPolicy::default()).unwrap().len() % 8,
            0
        ); // Should be multiple of alignment

        // Verify data was copied correctly
        let memory = controller.memory(AccessPolicy::default()).unwrap();
        let memory_slice =
            unsafe { core::slice::from_raw_parts(memory.as_ptr() as *const u8, data.len()) };
        assert_eq!(memory_slice, data);
    }

    #[test_log::test]
    fn test_core_allocation_controller_from_elems() {
        let elems = vec![1u32, 2, 3, 4];
        let expected_bytes = elems.len() * core::mem::size_of::<u32>();

        let controller = NativeAllocationController::from_elems(elems);
        assert_eq!(controller.alloc_align(), core::mem::align_of::<u32>());
        assert_eq!(
            controller.memory(AccessPolicy::default()).unwrap().len(),
            expected_bytes
        );
    }

    #[test_log::test]
    fn test_core_allocation_controller_grow() {
        let mut controller =
            NativeAllocationController::alloc_with_capacity(Global, 32, 8).unwrap();
        let old_memory_len = controller.memory(AccessPolicy::default()).unwrap().len();

        controller.grow(64, 8).unwrap();

        assert_eq!(controller.alloc_align(), 8);
        assert!(controller.memory(AccessPolicy::default()).unwrap().len() >= 64);
        assert!(controller.memory(AccessPolicy::default()).unwrap().len() > old_memory_len);
    }

    #[test_log::test]
    fn test_alloc_zero_size() {
        let controller = NativeAllocationController::alloc_with_capacity(Global, 0, 8).unwrap();
        assert_eq!(controller.ptr.as_ptr().align_offset(8), 0);
    }

    #[test_log::test]
    fn test_grow_from_zero() {
        let mut controller = NativeAllocationController::alloc_with_capacity(Global, 0, 8).unwrap();
        controller.grow(64, 8).unwrap();
        assert!(controller.capacity() >= 64);
        assert_eq!(controller.alloc_align(), 8);
    }

    #[test_log::test]
    fn test_grow_to_larger_alignment() {
        let mut controller =
            NativeAllocationController::alloc_with_data(Global, b"abcd", 1).unwrap();
        controller.grow(32, MAX_ALIGN).unwrap();
        assert_eq!(controller.alloc_align(), MAX_ALIGN);
        assert_eq!(controller.ptr.as_ptr().align_offset(MAX_ALIGN), 0);
        let memory = controller.memory(AccessPolicy::default()).unwrap();
        let memory = unsafe { core::slice::from_raw_parts(memory.as_ptr().cast::<u8>(), 4) };
        assert_eq!(memory, b"abcd");
    }

    #[test_log::test]
    fn test_memory_access() {
        let data = b"test data"; // 9 bytes, will be rounded up to 16 for 8-byte alignment
        let controller = NativeAllocationController::alloc_with_data(Global, data, 8).unwrap();

        let memory = controller.memory(AccessPolicy::default()).unwrap();
        assert!(memory.len() >= data.len());
        assert_eq!(memory.len() % 8, 0); // Should be multiple of alignment
        let memory_slice =
            unsafe { core::slice::from_raw_parts(memory.as_ptr() as *const u8, data.len()) };
        assert_eq!(memory_slice, data);
    }

    #[test_log::test]
    fn test_memory_mut_access() {
        let mut controller =
            NativeAllocationController::alloc_with_capacity(Global, 16, 8).unwrap();
        unsafe {
            let memory = controller.memory_mut(AccessPolicy::default()).unwrap();
            assert_eq!(memory.len(), 16);
            // Write some test data
            memory[0].write(42);
            memory[1].write(84);
        }
        // Verify the data was written
        let memory = controller.memory(AccessPolicy::default()).unwrap();
        unsafe {
            assert_eq!(memory[0].assume_init(), 42);
            assert_eq!(memory[1].assume_init(), 84);
        }
    }

    #[test_log::test]
    #[should_panic(expected = "capacity must be a multiple of alignment")]
    fn test_debug_assert_capacity_alignment_mismatch() {
        let _ = NativeAllocationController::alloc_with_capacity(Global, 33, 8);
    }
}
