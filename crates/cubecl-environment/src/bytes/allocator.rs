//! Memory that the application gives to [`Bytes`](super::Bytes).
//!
//! By default, [`Bytes`](super::Bytes) gets its memory from the global allocator. A
//! [`BufferAllocator`] lets the application put the data in a different memory, for example a
//! static arena or a fast SRAM region of a microcontroller.
//!
//! Each `Bytes` remembers the allocator that it came from. When `Bytes` must copy its data, for
//! example in `clone`, `split` or copy-on-write, the copy also comes from that allocator.

use super::{
    AllocationError, AllocationProperty,
    default_controller::{Controller, Provider},
};
use core::{
    alloc::{GlobalAlloc, Layout},
    ptr::NonNull,
};

/// A source of memory for [`Bytes`](super::Bytes).
///
/// The allocator must be `'static`. Then no `Bytes` can live longer than the memory that it
/// uses. An allocator that resets its memory must refuse to reset while allocations are live.
///
/// Use [`FromGlobalAlloc`] for a heap that implements [`GlobalAlloc`]. Most embedded heaps do.
///
/// # Safety
///
/// - [`allocate`](Self::allocate) must return memory that is valid for reads and writes of
///   `layout.size()` bytes, and that is aligned to `layout.align()`.
/// - The memory must stay valid until [`deallocate`](Self::deallocate) or
///   [`grow`](Self::grow) receives the pointer.
/// - Two live allocations must not overlap.
pub unsafe trait BufferAllocator: Send + Sync {
    /// Allocates memory for `layout`.
    ///
    /// The size of `layout` is never zero.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError::OutOfMemory`] when the allocator has no memory for `layout`.
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocationError>;

    /// Releases the memory at `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must come from [`allocate`](Self::allocate) or [`grow`](Self::grow) of this
    /// allocator, and `layout` must be the layout of that call.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Moves the allocation at `ptr` from layout `old` to the larger layout `new`.
    ///
    /// The first `old.size()` bytes keep their values. The alignment of `new` can be different
    /// from the alignment of `old`. The default allocates `new`, copies the bytes and releases
    /// `old`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError::OutOfMemory`] when the allocator has no memory for `new`.
    /// Then `ptr` stays valid with layout `old`.
    ///
    /// # Safety
    ///
    /// - `ptr` must come from this allocator, with layout `old`.
    /// - `new.size()` must be equal to or larger than `old.size()`.
    /// - On success, `ptr` is not valid anymore.
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<u8>, AllocationError> {
        // SAFETY: The caller gives the same guarantees that `grow_by_copy` requires.
        unsafe { grow_by_copy(self, ptr, old, new) }
    }
}

/// Allocates `new`, copies `old.size()` bytes from `ptr` and releases `ptr`.
///
/// # Safety
///
/// The same as [`BufferAllocator::grow`].
unsafe fn grow_by_copy<A: BufferAllocator + ?Sized>(
    allocator: &A,
    ptr: NonNull<u8>,
    old: Layout,
    new: Layout,
) -> Result<NonNull<u8>, AllocationError> {
    let new_ptr = allocator.allocate(new)?;
    // SAFETY: The two allocations are different and both have at least `old.size()` bytes.
    // `ptr` comes from `allocator` with layout `old`.
    unsafe {
        core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), old.size());
        allocator.deallocate(ptr, old);
    }
    Ok(new_ptr)
}

/// Uses a [`GlobalAlloc`] as a [`BufferAllocator`].
///
/// With this adapter, a heap can give memory to [`Bytes`](super::Bytes) without being the
/// `#[global_allocator]` of the program.
///
/// ```ignore
/// static ARENA: FromGlobalAlloc<embedded_alloc::LlffHeap> =
///     FromGlobalAlloc(embedded_alloc::LlffHeap::empty());
///
/// let bytes = Bytes::zeroed_in(&ARENA, 1024, 4)?;
/// ```
#[derive(Debug, Default)]
pub struct FromGlobalAlloc<A>(pub A);

// SAFETY: `GlobalAlloc` gives the same guarantees as `BufferAllocator` for a layout of non-zero
// size, and a null pointer becomes an error.
unsafe impl<A: GlobalAlloc + Send + Sync> BufferAllocator for FromGlobalAlloc<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocationError> {
        // SAFETY: `BufferAllocator::allocate` never receives a layout of zero size.
        NonNull::new(unsafe { self.0.alloc(layout) }).ok_or(AllocationError::OutOfMemory)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: The caller gives the pointer and the layout of an earlier allocation.
        unsafe { self.0.dealloc(ptr.as_ptr(), layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<u8>, AllocationError> {
        // `GlobalAlloc::realloc` cannot change the alignment. Copy for a new alignment.
        if old.align() != new.align() {
            // SAFETY: The caller gives the guarantees of `BufferAllocator::grow`.
            return unsafe { grow_by_copy(self, ptr, old, new) };
        }
        // SAFETY: `ptr` comes from this allocator with layout `old`, and `new.size()` is not
        // zero because it is equal to or larger than `old.size()`.
        let ptr = unsafe { self.0.realloc(ptr.as_ptr(), old, new.size()) };
        NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)
    }
}

/// Uses an [`allocator_api2::alloc::Allocator`] as a [`BufferAllocator`].
#[cfg(feature = "allocator-api2")]
#[derive(Debug, Default)]
pub struct FromAllocatorApi2<A>(pub A);

// SAFETY: `Allocator` gives the same guarantees as `BufferAllocator`. `BufferAllocator` always
// deallocates with the layout of the allocation, and that layout fits the returned block.
#[cfg(feature = "allocator-api2")]
unsafe impl<A: allocator_api2::alloc::Allocator + Send + Sync> BufferAllocator
    for FromAllocatorApi2<A>
{
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocationError> {
        self.0
            .allocate(layout)
            .map(NonNull::cast)
            .map_err(|_| AllocationError::OutOfMemory)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: The caller gives the pointer and the layout of an earlier allocation.
        unsafe { self.0.deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<u8>, AllocationError> {
        // SAFETY: The caller gives the guarantees that `Allocator::grow` requires.
        unsafe { self.0.grow(ptr, old, new) }
            .map(NonNull::cast)
            .map_err(|_| AllocationError::OutOfMemory)
    }
}

// SAFETY: A shared reference gives the guarantees of the allocator that it points to.
unsafe impl<A: BufferAllocator + ?Sized> BufferAllocator for &A {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocationError> {
        (**self).allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: The caller gives the guarantees of `BufferAllocator::deallocate`.
        unsafe { (**self).deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<u8>, AllocationError> {
        // SAFETY: The caller gives the guarantees of `BufferAllocator::grow`.
        unsafe { (**self).grow(ptr, old, new) }
    }
}

// Memory from an allocator of the application never becomes a `Vec`, because a `Vec` releases
// its memory with the global allocator.
impl Provider for &'static dyn BufferAllocator {
    const DETACHABLE: bool = false;
    const PROPERTY: AllocationProperty = AllocationProperty::Other;
}

/// Allocation controller whose memory comes from a [`BufferAllocator`].
pub(crate) type AllocatorController = Controller<&'static dyn BufferAllocator>;

// The tests need std (the system allocator and test_log).
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::bytes::{Bytes, ScratchBuffer, SplitPolicy, default_controller::MAX_ALIGN};
    use alloc::{boxed::Box, vec};
    use core::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(feature = "dyn-stack")]
    use {
        crate::bytes::DynStackAllocator,
        dyn_stack::{MemStack, StackReq, mem::MemBuffer},
    };

    /// Counts the calls that reach it, and refuses allocations above `limit` live bytes.
    struct Counting {
        inner: FromGlobalAlloc<std::alloc::System>,
        allocs: AtomicUsize,
        grows: AtomicUsize,
        frees: AtomicUsize,
        live: AtomicUsize,
        limit: usize,
    }

    impl Counting {
        /// A new allocator for one test. Each test has its own, so tests can run in parallel.
        fn leak(limit: usize) -> &'static Self {
            Box::leak(Box::new(Self {
                inner: FromGlobalAlloc(std::alloc::System),
                allocs: AtomicUsize::new(0),
                grows: AtomicUsize::new(0),
                frees: AtomicUsize::new(0),
                live: AtomicUsize::new(0),
                limit,
            }))
        }

        fn counts(&self) -> (usize, usize, usize) {
            (
                self.allocs.load(Ordering::Relaxed),
                self.grows.load(Ordering::Relaxed),
                self.frees.load(Ordering::Relaxed),
            )
        }

        fn live(&self) -> usize {
            self.live.load(Ordering::Relaxed)
        }

        fn reserve(&self, size: usize) -> Result<(), AllocationError> {
            if self.live() + size > self.limit {
                return Err(AllocationError::OutOfMemory);
            }
            self.live.fetch_add(size, Ordering::Relaxed);
            Ok(())
        }
    }

    unsafe impl BufferAllocator for Counting {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocationError> {
            self.reserve(layout.size())?;
            self.allocs.fetch_add(1, Ordering::Relaxed);
            self.inner.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.frees.fetch_add(1, Ordering::Relaxed);
            self.live.fetch_sub(layout.size(), Ordering::Relaxed);
            unsafe { self.inner.deallocate(ptr, layout) }
        }

        unsafe fn grow(
            &self,
            ptr: NonNull<u8>,
            old: Layout,
            new: Layout,
        ) -> Result<NonNull<u8>, AllocationError> {
            self.reserve(new.size() - old.size())?;
            self.grows.fetch_add(1, Ordering::Relaxed);
            unsafe { self.inner.grow(ptr, old, new) }
        }
    }

    #[test_log::test]
    fn zeroed_in_allocates_once_and_frees_on_drop() {
        let arena = Counting::leak(usize::MAX);
        let bytes = Bytes::zeroed_in(arena, 10, 4).unwrap();
        assert_eq!(&bytes[..], &[0; 10]);
        assert_eq!(bytes.capacity(), 12);
        assert_eq!(arena.counts(), (1, 0, 0));

        drop(bytes);
        assert_eq!(arena.counts(), (1, 0, 1));
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn zero_capacity_does_not_allocate() {
        let arena = Counting::leak(usize::MAX);
        let mut bytes = Bytes::with_capacity_in(arena, 0, 8).unwrap();
        assert_eq!(arena.counts(), (0, 0, 0));

        bytes.extend_from_byte_slice(&[1, 2, 3]);
        assert_eq!(&bytes[..], &[1, 2, 3]);
        assert_eq!(arena.counts(), (1, 0, 0));
        drop(bytes);
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn try_into_vec_never_returns_allocator_memory() {
        let arena = Counting::leak(usize::MAX);
        // Alignment 1 and a capacity of whole `u8`s: every other check of `try_into_vec` passes.
        let bytes = Bytes::from_slice_in(arena, &[1, 2, 3, 4], 1).unwrap();
        let bytes = bytes.try_into_vec::<u8>().unwrap_err();
        assert_eq!(&bytes[..], &[1, 2, 3, 4]);
        drop(bytes);
        assert_eq!(arena.counts(), (1, 0, 1));
    }

    #[test_log::test]
    fn clone_stays_in_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        let bytes = Bytes::from_slice_in(arena, &[1, 2, 3, 4], 4).unwrap();
        let copy = bytes.clone();
        assert_eq!(copy, bytes);
        assert_eq!(copy.property(), AllocationProperty::Other);
        assert_eq!(arena.counts(), (2, 0, 0));
        drop((bytes, copy));
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn extend_grows_in_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        let mut bytes = Bytes::from_slice_in(arena, &[1, 2, 3, 4], 4).unwrap();
        bytes.extend_from_byte_slice(&[5, 6, 7, 8, 9]);
        assert_eq!(&bytes[..], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // `extend_from_byte_slice` asks for `MAX_ALIGN`, so the data becomes more aligned.
        assert_eq!(bytes.align(), MAX_ALIGN);
        assert_eq!(arena.counts(), (1, 1, 0));
        drop(bytes);
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn owned_split_stays_in_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        let bytes = Bytes::from_slice_in(arena, &[1, 2, 3, 4], 2).unwrap();
        let (left, right) = bytes.split(2, SplitPolicy::Owned).unwrap();
        assert_eq!((&left[..], &right[..]), (&[1, 2][..], &[3, 4][..]));
        assert_eq!(left.align(), 2);
        // One for the original, which is released, and one for each half.
        assert_eq!(arena.counts(), (3, 0, 1));
        drop((left, right));
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn shared_split_copies_on_write_into_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        let bytes = Bytes::from_slice_in(arena, &[1, 2, 3, 4], 2).unwrap();
        let (mut left, right) = bytes.split(2, SplitPolicy::Shared).unwrap();
        assert_eq!(arena.counts(), (1, 0, 0));

        left[0] = 9;
        assert_eq!((&left[..], &right[..]), (&[9, 2][..], &[3, 4][..]));
        assert_eq!(arena.counts(), (2, 0, 0));
        drop((left, right));
        assert_eq!(arena.live(), 0);
    }

    /// Read-only data, as weights in flash.
    #[cfg(feature = "shared-bytes")]
    static WEIGHTS: [u8; 4] = [1, 2, 3, 4];

    #[cfg(feature = "shared-bytes")]
    #[test_log::test]
    fn shared_bytes_copy_on_write_into_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        let bytes = Bytes::from_shared_in(
            bytes::Bytes::from_static(&WEIGHTS),
            AllocationProperty::Other,
            arena,
        );

        // Reads, clones and splits don't copy.
        let (left, right) = bytes.clone().split(2, SplitPolicy::Shared).unwrap();
        assert_eq!((&left[..], &right[..]), (&[1, 2][..], &[3, 4][..]));
        // `try_into_vec` fails without a copy.
        let mut bytes = bytes.try_into_vec::<u8>().unwrap_err();
        assert_eq!(arena.counts(), (0, 0, 0));

        // The first write copies into the allocator, and so do writes to the halves.
        bytes[0] = 9;
        let mut right = right;
        right[0] = 7;
        assert_eq!(&bytes[..], &[9, 2, 3, 4]);
        assert_eq!(&right[..], &[7, 4]);
        assert_eq!(WEIGHTS, [1, 2, 3, 4]);
        assert_eq!(arena.counts(), (2, 0, 0));
        drop((bytes, left, right));
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn exhausted_allocator_returns_an_error() {
        let arena = Counting::leak(16);
        let bytes = Bytes::zeroed_in(arena, 16, 4).unwrap();
        assert_eq!(
            Bytes::zeroed_in(arena, 1, 1).unwrap_err(),
            AllocationError::OutOfMemory
        );
        drop(bytes);
        Bytes::zeroed_in(arena, 16, 4).unwrap();
    }

    #[test_log::test]
    #[should_panic(expected = "failed to clone 16 bytes")]
    fn clone_panics_when_the_allocator_is_exhausted() {
        let arena = Counting::leak(16);
        let bytes = Bytes::zeroed_in(arena, 16, 4).unwrap();
        let _ = bytes.clone();
    }

    #[test_log::test]
    fn alignment_above_the_maximum_is_unsupported() {
        let arena = Counting::leak(usize::MAX);
        assert_eq!(
            Bytes::zeroed_in(arena, 64, 2 * MAX_ALIGN).unwrap_err(),
            AllocationError::UnsupportedOperation
        );
        assert_eq!(arena.counts(), (0, 0, 0));
    }

    #[test_log::test]
    fn global_alloc_adapter_grows_to_a_new_alignment() {
        let arena = FromGlobalAlloc(std::alloc::System);
        let old = Layout::from_size_align(4, 1).unwrap();
        let new = Layout::from_size_align(32, 16).unwrap();
        unsafe {
            let ptr = arena.allocate(old).unwrap();
            ptr.as_ptr()
                .copy_from_nonoverlapping([1, 2, 3, 4].as_ptr(), 4);
            let ptr = arena.grow(ptr, old, new).unwrap();
            assert_eq!(ptr.as_ptr().align_offset(16), 0);
            assert_eq!(core::slice::from_raw_parts(ptr.as_ptr(), 4), &[1, 2, 3, 4]);
            arena.deallocate(ptr, new);
        }
    }

    #[test_log::test]
    fn from_fn_in_writes_each_element_into_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        let bytes = Bytes::from_fn_in(arena, 5, |i| u32::try_from(i * 10).unwrap()).unwrap();
        assert_eq!(bytes.align(), align_of::<u32>());
        assert_eq!(bytes.len(), 20);
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&bytes),
            &[0, 10, 20, 30, 40]
        );
        assert_eq!(arena.counts(), (1, 0, 0));
        let bytes = bytes.try_into_vec::<u32>().unwrap_err();
        drop(bytes);
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn from_fn_without_an_allocator_becomes_a_vec_without_a_copy() {
        let bytes = Bytes::from_fn(4, |i| f32::from(u8::try_from(i).unwrap()));
        let ptr = bytes.as_ptr();
        let vec = bytes.try_into_vec::<f32>().unwrap();
        assert_eq!(vec, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(vec.as_ptr().cast::<u8>(), ptr);
    }

    #[test_log::test]
    fn output_like_an_input_stays_in_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        let input = Bytes::from_slice_in(arena, &[1, 2, 3, 4], 1).unwrap();
        let output = input
            .from_fn_like(3, |i| u16::try_from(i).unwrap())
            .unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u16>(&output), &[0, 1, 2]);
        assert_eq!(output.align(), align_of::<u16>());
        assert_eq!(output.property(), AllocationProperty::Other);
        assert_eq!(arena.counts(), (2, 0, 0));
        drop((input, output));
        assert_eq!(arena.live(), 0);
    }

    #[test_log::test]
    fn output_like_an_exhausted_allocator_is_an_error() {
        let arena = Counting::leak(4);
        let input = Bytes::from_slice_in(arena, &[1, 2, 3, 4], 1).unwrap();
        assert_eq!(
            input.with_capacity_like(1, 1).unwrap_err(),
            AllocationError::OutOfMemory
        );
    }

    #[test_log::test]
    fn output_like_global_memory_becomes_a_vec_without_a_copy() {
        let input = Bytes::from_elems(vec![1u8, 2, 3]);
        let output = input
            .from_fn_like(2, |i| f32::from(u8::try_from(i).unwrap()))
            .unwrap();
        let ptr = output.as_ptr();
        let vec = output.try_into_vec::<f32>().unwrap();
        assert_eq!(vec, [0.0, 1.0]);
        assert_eq!(vec.as_ptr().cast::<u8>(), ptr);
    }

    #[test_log::test]
    fn spare_capacity_and_set_len_write_an_output_in_place() {
        let arena = Counting::leak(usize::MAX);
        let mut bytes = Bytes::with_capacity_in(arena, 6, 2).unwrap();
        let spare = bytes.spare_capacity_mut();
        assert_eq!(spare.len(), 6);
        // Write in any order, as a scatter or a gather does.
        for i in (0..6).rev() {
            spare[i].write(u8::try_from(i).unwrap());
        }
        // SAFETY: The first 6 bytes are initialized.
        unsafe { bytes.set_len(6) };
        assert_eq!(&bytes[..], &[0, 1, 2, 3, 4, 5]);
        assert!(bytes.spare_capacity_mut().is_empty());
        assert_eq!(arena.counts(), (1, 0, 0));
    }

    #[test_log::test]
    fn spare_capacity_of_global_memory_becomes_a_vec() {
        let mut bytes = Bytes::with_capacity(8, 4).unwrap();
        for (i, byte) in bytes.spare_capacity_mut().iter_mut().enumerate() {
            byte.write(u8::try_from(i).unwrap());
        }
        // SAFETY: The 8 bytes of capacity are initialized.
        unsafe { bytes.set_len(8) };
        // The alignment of the memory is the alignment of `u32`.
        let vec = bytes.try_into_vec::<u32>().unwrap();
        assert_eq!(
            vec,
            [
                u32::from_ne_bytes([0, 1, 2, 3]),
                u32::from_ne_bytes([4, 5, 6, 7])
            ]
        );
    }

    #[test_log::test]
    fn scratch_buffer_has_any_alignment_and_frees_on_drop() {
        let arena = Counting::leak(usize::MAX);
        let layout = Layout::from_size_align(256, 64).unwrap();
        let mut scratch = ScratchBuffer::new_in(arena, layout).unwrap();
        let memory = scratch.as_uninit_mut();
        assert_eq!(memory.len(), 256);
        assert_eq!(memory.as_ptr().align_offset(64), 0);
        assert_eq!(scratch.layout(), layout);
        assert_eq!(arena.counts(), (1, 0, 0));
        drop(scratch);
        assert_eq!(arena.counts(), (1, 0, 1));

        let empty = ScratchBuffer::new_in(arena, Layout::from_size_align(0, 64).unwrap());
        assert!(empty.unwrap().as_uninit_mut().is_empty());
        assert_eq!(arena.counts(), (1, 0, 1));

        let mut global = ScratchBuffer::new(layout);
        assert_eq!(global.as_uninit_mut().as_ptr().align_offset(64), 0);
    }

    #[test_log::test]
    fn scratch_buffer_reports_an_exhausted_allocator() {
        let arena = Counting::leak(16);
        let layout = Layout::from_size_align(32, 8).unwrap();
        assert_eq!(
            ScratchBuffer::new_in(arena, layout).unwrap_err(),
            AllocationError::OutOfMemory
        );
    }

    #[cfg(feature = "dyn-stack")]
    #[test_log::test]
    fn dyn_stack_buffers_come_from_the_allocator() {
        let arena = Counting::leak(usize::MAX);
        // The packing buffers of a matrix multiplication, aligned for SIMD.
        let req = StackReq::new_aligned::<f32>(100, 64).and(StackReq::new_aligned::<f32>(50, 64));
        let mut buffer = MemBuffer::new_in(req, DynStackAllocator(arena as &dyn BufferAllocator));
        assert_eq!(arena.counts(), (1, 0, 0));

        let stack = MemStack::new(&mut buffer);
        let (lhs, stack) = stack.make_aligned_uninit::<f32>(100, 64);
        let (rhs, _) = stack.make_aligned_uninit::<f32>(50, 64);
        assert_eq!((lhs.len(), rhs.len()), (100, 50));
        assert_eq!(lhs.as_ptr().align_offset(64), 0);
        drop(buffer);
        assert_eq!(arena.counts(), (1, 0, 1));
        assert_eq!(arena.live(), 0);
    }

    #[cfg(feature = "dyn-stack")]
    #[test_log::test]
    fn a_scratch_buffer_can_back_a_dyn_stack() {
        let arena = Counting::leak(usize::MAX);
        let req = StackReq::new_aligned::<f32>(64, 64);
        let mut scratch = ScratchBuffer::new_in(arena, req.layout().unwrap()).unwrap();
        // Use the same buffer for each call: only one allocation.
        for _ in 0..3 {
            let stack = MemStack::new(scratch.as_uninit_mut());
            let (packed, _) =
                stack.make_aligned_with::<f32>(64, 64, |i| f32::from(u8::try_from(i).unwrap()));
            assert_eq!(packed[63], 63.0);
        }
        assert_eq!(arena.counts(), (1, 0, 0));
    }
}
