//! Allocation controller that shares another [`Bytes`] through an [`Arc`].
//!
//! Unlike the other controllers, [`SharedAllocationController`] does not own an
//! allocation directly. Instead it composes over an existing [`Bytes`] held
//! behind an [`Arc`], optionally viewing only a sub-range of it. This makes
//! cloning and splitting cheap (reference counted, zero-copy) at the cost of
//! never being able to detach the allocation into a `Vec` (so
//! [`Bytes::try_into_vec`](super::Bytes::try_into_vec) always fails on shared
//! bytes).
//!
//! # Example
//!
//! ```
//! use cubecl_common::bytes::Bytes;
//!
//! let bytes = Bytes::from_elems(vec![1u32, 2, 3, 4]);
//! let shared = bytes.shared();
//! // Cloning is cheap, both clones reference the same backing allocation.
//! let clone = shared.clone();
//! assert_eq!(&shared[..], &clone[..]);
//! ```

use super::{
    AllocationController, AllocationProperty, Bytes, SplitError,
    default_controller::{MAX_ALIGN, NativeAllocationController},
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};

/// Allocation controller that shares a view into another [`Bytes`] behind an [`Arc`].
///
/// # Safety
///
/// Like the file and `bytes::Bytes` controllers, this uses an [`UnsafeCell`] to
/// lazily copy the shared content into a private buffer on first mutable access
/// (copy-on-write). This is sound because mutable access requires `&mut self`,
/// which is exclusive, and the private buffer is only ever published once.
pub struct SharedAllocationController {
    /// The shared underlying bytes.
    inner: Arc<Bytes>,
    /// Offset, in bytes, of this view into `inner`.
    offset: usize,
    /// Length, in bytes, of this view.
    len: usize,
    /// Lazily initialized private buffer (copy-on-write).
    controller: UnsafeCell<Option<Box<dyn AllocationController>>>,
    /// Whether the private buffer has been initialized (data copied out).
    init: AtomicBool,
}

impl SharedAllocationController {
    /// Create a controller viewing `inner[offset..offset + len]`.
    pub(crate) fn new(inner: Arc<Bytes>, offset: usize, len: usize) -> Self {
        debug_assert!(
            offset + len <= inner.len(),
            "shared view must stay within the bounds of the inner bytes"
        );
        Self {
            inner,
            offset,
            len,
            controller: UnsafeCell::new(None),
            init: AtomicBool::new(false),
        }
    }

    /// The shared view, valid as long as no copy-on-write has occurred.
    fn view(&self) -> &[u8] {
        &self.inner[self.offset..self.offset + self.len]
    }

    /// Copy the shared view into a private, writable native allocation.
    /// Called lazily on first mutable access (copy-on-write).
    fn init_mutable(&self) {
        if self.init.load(Ordering::Relaxed) {
            return;
        }

        // Allocate with `MAX_ALIGN` to keep the data usable for any element type.
        let controller = NativeAllocationController::alloc_with_data(self.view(), MAX_ALIGN)
            .expect("failed to allocate copy-on-write buffer for shared bytes");

        // SAFETY: We only write to the `UnsafeCell` while `init` is false, and
        // mutable access requires `&mut self`, so no concurrent access exists.
        unsafe {
            *self.controller.get() = Some(Box::new(controller));
        }
        self.init.store(true, Ordering::Relaxed);
    }
}

impl AllocationController for SharedAllocationController {
    fn alloc_align(&self) -> usize {
        if self.init.load(Ordering::Relaxed) {
            MAX_ALIGN
        } else {
            // Report the inner allocation's alignment. `try_into_vec` still
            // fails because `try_detach` is never implemented for shared bytes.
            self.inner.align()
        }
    }

    fn property(&self) -> AllocationProperty {
        self.inner.property()
    }

    fn memory(&self) -> &[MaybeUninit<u8>] {
        if self.init.load(Ordering::Relaxed) {
            // SAFETY: `init` is true, so the private controller is `Some`.
            unsafe {
                (*self.controller.get())
                    .as_ref()
                    .expect("controller must be Some when init is true")
                    .memory()
            }
        } else {
            let slice = self.view();
            // SAFETY: `&[u8]` and `&[MaybeUninit<u8>]` share a layout, and every
            // byte of the shared view is initialized.
            unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
        }
    }

    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // Trigger copy-on-write so we never mutate the shared allocation.
        self.init_mutable();

        // SAFETY: `init_mutable` guarantees the private controller is `Some`.
        unsafe {
            (*self.controller.get())
                .as_mut()
                .expect("controller must be Some after init_mutable")
                .memory_mut()
        }
    }

    fn split(
        &mut self,
        offset: usize,
    ) -> Result<(Box<dyn AllocationController>, Box<dyn AllocationController>), SplitError> {
        if self.init.load(Ordering::Relaxed) {
            // After copy-on-write the private buffer is no longer shared.
            return Err(SplitError::Unsupported);
        }
        // Use `>` (not `>=`) to allow boundary splits where one side is empty.
        if offset > self.len {
            return Err(SplitError::InvalidOffset);
        }

        let left = SharedAllocationController::new(self.inner.clone(), self.offset, offset);
        let right = SharedAllocationController::new(
            self.inner.clone(),
            self.offset + offset,
            self.len - offset,
        );

        Ok((Box::new(left), Box::new(right)))
    }

    fn view(&self, start: usize, end: usize) -> Option<Box<dyn AllocationController>> {
        if self.init.load(Ordering::Relaxed) {
            // After copy-on-write the private buffer is no longer shared.
            return None;
        }
        if start > end || end > self.len {
            return None;
        }

        Some(Box::new(SharedAllocationController::new(
            self.inner.clone(),
            self.offset + start,
            end - start,
        )))
    }

    fn duplicate(&self) -> Option<Box<dyn AllocationController>> {
        if self.init.load(Ordering::Relaxed) {
            // After mutation the private buffer can't be shared cheaply.
            return None;
        }

        Some(Box::new(SharedAllocationController::new(
            self.inner.clone(),
            self.offset,
            self.len,
        )))
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        if self.init.load(Ordering::Relaxed) {
            let memory = self.memory();
            let copy_len = buf.len().min(memory.len());
            // SAFETY: every byte of the private buffer up to its length is initialized.
            let data =
                unsafe { core::slice::from_raw_parts(memory.as_ptr().cast::<u8>(), copy_len) };
            buf[..copy_len].copy_from_slice(data);
        } else {
            let src = self.view();
            let copy_len = buf.len().min(src.len());
            buf[..copy_len].copy_from_slice(&src[..copy_len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Bytes, SplitPolicy};
    use alloc::vec;

    #[test_log::test]
    fn test_shared_is_zero_copy_view() {
        let bytes = Bytes::from_elems(vec![1u8, 2, 3, 4]);
        let shared = bytes.shared();
        assert_eq!(&shared[..], &[1, 2, 3, 4]);
        assert_eq!(shared.len(), 4);
    }

    #[test_log::test]
    fn test_shared_clone_is_cheap_and_equal() {
        let shared = Bytes::from_elems(vec![10u32, 20, 30]).shared();
        let clone = shared.clone();
        assert_eq!(&shared[..], &clone[..]);
    }

    #[test_log::test]
    fn test_shared_try_into_vec_never_succeeds() {
        let shared = Bytes::from_elems(vec![1u8, 2, 3, 4]).shared();
        assert!(shared.try_into_vec::<u8>().is_err());
    }

    #[test_log::test]
    fn test_shared_split() {
        let shared = Bytes::from_elems(vec![0u8, 1, 2, 3, 4, 5, 6, 7]).shared();
        let (left, right) = shared.split(3, SplitPolicy::Shared).unwrap();
        assert_eq!(&left[..], &[0, 1, 2]);
        assert_eq!(&right[..], &[3, 4, 5, 6, 7]);
    }

    #[test_log::test]
    fn test_shared_split_then_clone() {
        let shared = Bytes::from_elems(vec![0u8, 1, 2, 3, 4, 5]).shared();
        let (left, right) = shared.split(2, SplitPolicy::Shared).unwrap();
        assert_eq!(&left.clone()[..], &[0, 1]);
        assert_eq!(&right.clone()[..], &[2, 3, 4, 5]);
    }

    #[test_log::test]
    fn test_shared_view_is_zero_copy() {
        let shared = Bytes::from_elems(vec![0u8, 1, 2, 3, 4, 5]).shared();
        let view = shared.view(1, 4, SplitPolicy::Shared).unwrap();
        assert_eq!(&view[..], &[1, 2, 3]);
        // The original is untouched and the window can't be detached into a Vec.
        assert_eq!(&shared[..], &[0, 1, 2, 3, 4, 5]);
        assert!(view.try_into_vec::<u8>().is_err());
    }

    #[test_log::test]
    fn test_shared_copy_on_write() {
        let shared = Bytes::from_elems(vec![1u8, 2, 3, 4]).shared();
        let mut clone = shared.clone();
        clone[0] = 99;

        // The mutation is private to the clone, the original is untouched.
        assert_eq!(&clone[..], &[99, 2, 3, 4]);
        assert_eq!(&shared[..], &[1, 2, 3, 4]);
    }

    #[test_log::test]
    fn test_shared_extend() {
        let mut shared = Bytes::from_elems(vec![1u8, 2, 3]).shared();
        shared.extend_from_byte_slice(&[4, 5, 6]);
        assert_eq!(&shared[..], &[1, 2, 3, 4, 5, 6]);
    }
}
