//! Scratch memory for kernels, from a [`BufferAllocator`].
//!
//! Kernels such as a matrix multiplication need temporary memory: packing buffers, SIMD
//! buffers and FFT workspace. [`ScratchBuffer`] gets this memory from the same allocator as the
//! tensor data. Unlike [`Bytes`](super::Bytes), it has no limit on the alignment, and it does not
//! initialize its memory.
//!
//! With the `dyn-stack` feature, [`DynStackAllocator`] gives a [`BufferAllocator`] to the
//! `dyn-stack` crate. Libraries that use `dyn-stack` for their scratch memory, such as `gemm`,
//! can then allocate from it.

use super::{
    AllocationError, BufferAllocator,
    default_controller::{GLOBAL, dangling},
};
use core::{alloc::Layout, mem::MaybeUninit, ptr::NonNull};

/// Uninitialized memory for temporary data, from a [`BufferAllocator`].
///
/// The memory is released when the buffer is dropped. Allocate the buffer once and use it again
/// for each call, so that a kernel does not allocate per call.
///
/// ```ignore
/// let req = gemm_scratch_req(m, n, k);
/// let mut scratch = ScratchBuffer::new_in(&ARENA, req.layout()?)?;
/// gemm_with_stack(dst, lhs, rhs, MemStack::new(scratch.as_uninit_mut()));
/// ```
pub struct ScratchBuffer {
    allocator: &'static dyn BufferAllocator,
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: The buffer owns its memory, and `BufferAllocator` is `Send + Sync`.
unsafe impl Send for ScratchBuffer {}
// SAFETY: Only `&mut self` gives access to the memory.
unsafe impl Sync for ScratchBuffer {}

impl ScratchBuffer {
    /// Allocates memory for `layout` from `allocator`.
    ///
    /// A layout of zero size does not allocate.
    ///
    /// # Errors
    ///
    /// Returns [`AllocationError::OutOfMemory`] when `allocator` has no memory for `layout`.
    pub fn new_in(
        allocator: &'static dyn BufferAllocator,
        layout: Layout,
    ) -> Result<Self, AllocationError> {
        let ptr = if layout.size() == 0 {
            dangling(layout.align())
        } else {
            allocator.allocate(layout)?
        };
        Ok(Self {
            allocator,
            ptr,
            layout,
        })
    }

    /// Allocates memory for `layout` from the global allocator.
    ///
    /// When the global allocator has no memory, the program aborts, as with `Vec`.
    #[must_use]
    pub fn new(layout: Layout) -> Self {
        match Self::new_in(&GLOBAL, layout) {
            Ok(buffer) => buffer,
            Err(_) => unreachable!("the global allocator aborts instead of returning an error"),
        }
    }

    /// The memory of the buffer.
    ///
    /// The memory is not initialized when the buffer is created. Values from an earlier use stay
    /// in the memory.
    pub fn as_uninit_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: `ptr` is valid for `layout.size()` bytes, and `&mut self` is exclusive.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.layout.size()) }
    }

    /// The size and the alignment of the buffer.
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: `ptr` comes from `allocator` with this layout.
            unsafe { self.allocator.deallocate(self.ptr, self.layout) }
        }
    }
}

impl core::fmt::Debug for ScratchBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ScratchBuffer")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// Uses a [`BufferAllocator`] as a [`dyn_stack::alloc::Allocator`].
///
/// Then `dyn_stack::MemBuffer::new_in` and `dyn_stack::PodBuffer::new_in` get their memory from
/// the [`BufferAllocator`]. Use `DynStackAllocator(&ARENA)` for a `static` allocator.
#[cfg(feature = "dyn-stack")]
#[derive(Debug, Default, Clone, Copy)]
pub struct DynStackAllocator<A>(pub A);

// SAFETY: `BufferAllocator` gives the same guarantees for a layout of non-zero size. A layout of
// zero size gets a dangling pointer and is never released.
#[cfg(feature = "dyn-stack")]
unsafe impl<A: BufferAllocator> dyn_stack::alloc::Allocator for DynStackAllocator<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, dyn_stack::alloc::AllocError> {
        let ptr = if layout.size() == 0 {
            dangling(layout.align())
        } else {
            self.0
                .allocate(layout)
                .map_err(|_| dyn_stack::alloc::AllocError)?
        };
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: The caller gives the pointer and the layout of an earlier allocation.
            unsafe { self.0.deallocate(ptr, layout) }
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, dyn_stack::alloc::AllocError> {
        if old_layout.size() == 0 {
            return self.allocate(new_layout);
        }
        // SAFETY: The caller gives the guarantees of `BufferAllocator::grow`.
        let ptr = unsafe { self.0.grow(ptr, old_layout, new_layout) }
            .map_err(|_| dyn_stack::alloc::AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}
