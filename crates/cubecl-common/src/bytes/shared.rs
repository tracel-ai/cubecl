//! Shared bytes allocation controller for zero-copy tensor loading.
//!
//! This module provides [`SharedBytesAllocationController`] which implements the
//! [`AllocationController`] trait, enabling zero-copy tensor data access from
//! [`bytes::Bytes`] buffers.
//!
//! # Use Cases
//!
//! - **Static embedded data**: Use [`bytes::Bytes::from_static`] to reference data
//!   embedded in the binary via `include_bytes!` without heap allocation.
//! - **Memory-mapped files**: Use [`bytes::Bytes::from_owner`] to wrap memory-mapped
//!   regions while keeping the mmap alive through Arc reference counting.
//!
//! # Example
//!
//! ```
//! use cubecl_common::bytes::Bytes;
//!
//! // Zero-copy from static data
//! static DATA: &[u8] = &[1, 2, 3, 4];
//! let shared = bytes::Bytes::from_static(DATA);
//! let bytes = Bytes::from_shared(shared);
//! ```

use super::{
    AllocationController, AllocationError, AllocationProperty, SplitError,
    default_controller::NativeAllocationController,
};
use alloc::boxed::Box;
use alloc::vec;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, Ordering};

/// Allocation controller backed by [`bytes::Bytes`] for zero-copy access.
///
/// This enables zero-copy tensor loading by referencing data directly from
/// the source buffer without copying into heap-allocated memory.
///
/// # Safety
///
/// This implementation uses an [`UnsafeCell`] to lazily copy the content into an in-memory
/// buffer using [`NativeAllocationController`] when mutable access is requested. It is safe
/// because the controller can't be cloned or synced between multiple threads. You can
/// duplicate the shared bytes allocator, but every version of it will have its own buffer.
///
/// # Notes
///
/// - Read access via [`memory()`](AllocationController::memory) is zero-copy as long as no
///   mutable access has been requested.
/// - Mutable access via [`memory_mut()`](AllocationController::memory_mut) triggers a copy
///   of the data into heap memory (copy-on-write semantics).
/// - The underlying [`bytes::Bytes`] remains valid for the lifetime of this controller.
///   This is ensured by [`bytes::Bytes`] using reference counting internally.
pub struct SharedBytesAllocationController {
    /// The backing bytes buffer.
    bytes: bytes::Bytes,
    /// Lazily initialized mutable controller (copy-on-write).
    controller: UnsafeCell<Option<Box<dyn AllocationController>>>,
    /// Whether the controller has been initialized (data copied to heap).
    init: AtomicBool,
}

impl SharedBytesAllocationController {
    /// Create a new controller from a [`bytes::Bytes`] buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use cubecl_common::bytes::SharedBytesAllocationController;
    ///
    /// // From static data (zero-copy, no heap allocation)
    /// let static_bytes = bytes::Bytes::from_static(&[1, 2, 3, 4]);
    /// let controller = SharedBytesAllocationController::new(static_bytes);
    ///
    /// // From owned data
    /// let owned_bytes = bytes::Bytes::from(vec![1, 2, 3, 4]);
    /// let controller = SharedBytesAllocationController::new(owned_bytes);
    /// ```
    pub fn new(bytes: bytes::Bytes) -> Self {
        Self {
            bytes,
            controller: UnsafeCell::new(None),
            init: AtomicBool::new(false),
        }
    }

    /// Copy the shared bytes into a mutable native allocation controller.
    /// This is called lazily on first mutable access (copy-on-write).
    fn init_mutable(&self) {
        if self.init.load(Ordering::Relaxed) {
            return;
        }

        let data: &[u8] = &self.bytes;
        let mut buf = vec![0u8; data.len()];
        buf.copy_from_slice(data);

        let controller = NativeAllocationController::from_elems(buf);
        // SAFETY: We only write to the UnsafeCell when init is false,
        // and set init to true immediately after. This is safe because
        // UnsafeCell makes this type !Sync, preventing concurrent access.
        // The atomic bool provides an efficient check without locking.
        unsafe {
            *self.controller.get() = Some(Box::new(controller));
        }
        self.init.store(true, Ordering::Relaxed);
    }
}

impl AllocationController for SharedBytesAllocationController {
    fn alloc_align(&self) -> usize {
        if self.init.load(Ordering::Relaxed) {
            // After copy-on-write, use the native controller's alignment
            unsafe {
                (*self.controller.get())
                    .as_ref()
                    .map(|c| c.alloc_align())
                    .unwrap_or(1)
            }
        } else {
            // bytes::Bytes doesn't guarantee any particular alignment.
            // Report byte alignment (1) since we support arbitrary offsets.
            1
        }
    }

    fn property(&self) -> AllocationProperty {
        // bytes::Bytes could be backed by static data, heap, or mmap (via from_owner).
        // We report Other since we don't know the underlying storage type.
        AllocationProperty::Other
    }

    fn memory(&self) -> &[MaybeUninit<u8>] {
        if self.init.load(Ordering::Relaxed) {
            // Data has been copied to mutable controller, use that
            unsafe {
                (*self.controller.get())
                    .as_ref()
                    .map(|c| c.memory())
                    .unwrap_or(&[])
            }
        } else {
            // Zero-copy access to original bytes
            let slice: &[u8] = &self.bytes;
            // SAFETY: &[u8] and &[MaybeUninit<u8>] have the same memory layout,
            // and all bytes in the slice are initialized (coming from bytes::Bytes).
            unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
        }
    }

    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // Trigger copy-on-write if not already done
        self.init_mutable();

        // SAFETY: init_mutable() guarantees the controller is initialized
        unsafe {
            (*self.controller.get())
                .as_mut()
                .map(|c| c.memory_mut())
                .unwrap_or(&mut [])
        }
    }

    fn split(
        &mut self,
        offset: usize,
    ) -> Result<(Box<dyn AllocationController>, Box<dyn AllocationController>), SplitError> {
        if self.init.load(Ordering::Relaxed) {
            // Already copied to heap, can't split the original bytes anymore
            return Err(SplitError::Unsupported);
        }

        // Use `>` (not `>=`) to allow boundary splits where one side is empty.
        // This is symmetric: both offset==0 (empty left) and offset==len (empty right) are valid.
        if offset > self.bytes.len() {
            return Err(SplitError::InvalidOffset);
        }

        // bytes::Bytes supports efficient slicing (reference counted, no copy)
        let left = self.bytes.slice(..offset);
        let right = self.bytes.slice(offset..);

        Ok((
            Box::new(SharedBytesAllocationController::new(left)),
            Box::new(SharedBytesAllocationController::new(right)),
        ))
    }

    fn duplicate(&self) -> Option<Box<dyn AllocationController>> {
        if self.init.load(Ordering::Relaxed) {
            // After mutation, can't duplicate (forces full copy in Clone)
            return None;
        }

        // bytes::Bytes is Clone (reference counted), so duplication is cheap
        Some(Box::new(SharedBytesAllocationController::new(
            self.bytes.clone(),
        )))
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        if self.init.load(Ordering::Relaxed) {
            // Use the mutable controller's data
            let memory = self.memory();
            let copy_len = buf.len().min(memory.len());
            let memory_slice = &memory[..copy_len];
            // SAFETY: By construction, bytes are initialized.
            let data = unsafe {
                core::slice::from_raw_parts(memory_slice.as_ptr().cast(), memory_slice.len())
            };
            buf[..copy_len].copy_from_slice(data);
        } else {
            // Copy directly from shared bytes
            let src: &[u8] = &self.bytes;
            let copy_len = buf.len().min(src.len());
            buf[..copy_len].copy_from_slice(&src[..copy_len]);
        }
    }

    fn grow(&mut self, _size: usize, _align: usize) -> Result<(), AllocationError> {
        // Even after copy-on-write, we don't support growing
        // (would need to track original capacity vs NativeAllocationController's capacity)
        Err(AllocationError::UnsupportedOperation)
    }

    fn try_detach(&mut self) -> Option<NonNull<u8>> {
        // Memory is not managed by Rust's standard allocator in a way we can detach.
        // Even after copy-on-write, the NativeAllocationController is boxed inside us.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::super::Bytes;
    use super::*;

    #[test]
    fn test_from_static() {
        static DATA: &[u8] = &[1, 2, 3, 4, 5];
        let shared = bytes::Bytes::from_static(DATA);
        let bytes = Bytes::from_shared(shared);

        assert_eq!(&bytes[..], &[1, 2, 3, 4, 5]);
        assert_eq!(bytes.len(), 5);
    }

    #[test]
    fn test_from_vec() {
        let shared = bytes::Bytes::from(alloc::vec![10, 20, 30]);
        let bytes = Bytes::from_shared(shared);

        assert_eq!(&bytes[..], &[10, 20, 30]);
        assert_eq!(bytes.len(), 3);
    }

    #[test]
    fn test_split() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4, 5, 6]);
        let bytes = Bytes::from_shared(shared);

        let (left, right) = bytes.split(3).unwrap();

        assert_eq!(&left[..], &[1, 2, 3]);
        assert_eq!(&right[..], &[4, 5, 6]);
    }

    #[test]
    fn test_duplicate() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = SharedBytesAllocationController::new(shared);

        let dup = controller.duplicate().expect("duplicate should succeed");
        assert_eq!(dup.memory().len(), 3);
    }

    #[test]
    fn test_copy_into() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4, 5]);
        let controller = SharedBytesAllocationController::new(shared);

        let mut buf = [0u8; 3];
        unsafe { controller.copy_into(&mut buf) };
        assert_eq!(buf, [1, 2, 3]);
    }

    #[test]
    fn test_property() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = SharedBytesAllocationController::new(shared);

        assert!(matches!(controller.property(), AllocationProperty::Other));
    }

    #[test]
    fn test_alignment_before_mutation() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = SharedBytesAllocationController::new(shared);

        // Before mutation, alignment is 1 (no guarantee from bytes::Bytes)
        assert_eq!(controller.alloc_align(), 1);
    }

    #[test]
    fn test_grow_fails() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let mut controller = SharedBytesAllocationController::new(shared);

        let result = controller.grow(100, 1);
        assert!(matches!(result, Err(AllocationError::UnsupportedOperation)));
    }

    #[test]
    fn test_try_detach_returns_none() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let mut controller = SharedBytesAllocationController::new(shared);

        assert!(controller.try_detach().is_none());
    }

    #[test]
    fn test_try_into_vec_fails_gracefully() {
        // Since try_detach returns None, try_into_vec should fail and return the original Bytes
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4]);
        let bytes = Bytes::from_shared(shared);

        let result = bytes.try_into_vec::<u8>();
        assert!(result.is_err(), "try_into_vec should fail for shared bytes");

        // We should get the original bytes back
        let bytes = result.unwrap_err();
        assert_eq!(&bytes[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_copy_on_write() {
        // Test that mutable access triggers copy-on-write
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4, 5]);
        let mut bytes = Bytes::from_shared(shared);

        // Mutate the first byte
        bytes[0] = 99;

        // Verify the mutation worked
        assert_eq!(bytes[0], 99);
        assert_eq!(&bytes[1..], &[2, 3, 4, 5]);
    }

    #[test]
    fn test_clone_before_mutation_is_cheap() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let bytes = Bytes::from_shared(shared);

        // Clone should use duplicate() which is cheap (reference counted)
        let cloned = bytes.clone();

        assert_eq!(&bytes[..], &cloned[..]);
    }

    #[test]
    fn test_clone_after_mutation_copies() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let mut bytes = Bytes::from_shared(shared);

        // Trigger copy-on-write
        bytes[0] = 99;

        // Clone after mutation should do a full copy
        let cloned = bytes.clone();

        assert_eq!(&bytes[..], &cloned[..]);
        assert_eq!(cloned[0], 99);
    }

    #[test]
    fn test_slices_from_static_region() {
        // Simulate a static embedded data region (e.g., from include_bytes!)
        static EMBEDDED_DATA: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Create a shared bytes from the static region
        let shared = bytes::Bytes::from_static(EMBEDDED_DATA);

        // Slice into two parts: first 4 bytes and remaining 6 bytes
        let first_4 = shared.slice(0..4);
        let last_6 = shared.slice(4..10);

        // Create cubecl Bytes from each slice (zero-copy)
        let bytes_first = Bytes::from_shared(first_4);
        let bytes_last = Bytes::from_shared(last_6);

        // Verify contents
        assert_eq!(&bytes_first[..], &[1, 2, 3, 4]);
        assert_eq!(&bytes_last[..], &[5, 6, 7, 8, 9, 10]);

        // Verify lengths
        assert_eq!(bytes_first.len(), 4);
        assert_eq!(bytes_last.len(), 6);

        // Both should report alignment of 1 (no mutation yet)
        assert_eq!(bytes_first.align(), 1);
        assert_eq!(bytes_last.align(), 1);
    }

    #[test]
    fn test_multiple_slices_share_underlying_data() {
        // Demonstrate that multiple slices can be created from the same static region
        // without copying, and they all remain valid
        static DATA: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

        let shared = bytes::Bytes::from_static(DATA);

        // Create multiple overlapping and non-overlapping slices
        let slice_a = Bytes::from_shared(shared.slice(0..4)); // DEADBEEF
        let slice_b = Bytes::from_shared(shared.slice(4..8)); // CAFEBABE
        let slice_c = Bytes::from_shared(shared.slice(2..6)); // BEEF CAFE (overlapping)

        assert_eq!(&slice_a[..], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(&slice_b[..], &[0xCA, 0xFE, 0xBA, 0xBE]);
        assert_eq!(&slice_c[..], &[0xBE, 0xEF, 0xCA, 0xFE]);
    }
}
