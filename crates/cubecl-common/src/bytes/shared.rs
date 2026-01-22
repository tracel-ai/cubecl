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
//! - **Memory-mapped files**: Wrap memory-mapped regions in `bytes::Bytes` to keep
//!   the mapping alive through reference counting.
//!
//! # Example
//!
//! ```
//! use cubecl_common::bytes::{Bytes, AllocationProperty};
//!
//! // Zero-copy from static data
//! static DATA: &[u8] = &[1, 2, 3, 4];
//! let shared = bytes::Bytes::from_static(DATA);
//! let bytes = Bytes::from_shared(shared, AllocationProperty::Other);
//! ```

use super::{
    AllocationController, AllocationError, AllocationProperty, SplitError,
    default_controller::{MAX_ALIGN, NativeAllocationController},
};
use alloc::boxed::Box;
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
    /// The allocation property (used to optimize GPU transfers).
    property: AllocationProperty,
}

impl SharedBytesAllocationController {
    /// Create a new controller from a [`bytes::Bytes`] buffer with a specific
    /// allocation property.
    ///
    /// The allocation property is used by GPU backends to optimize data transfers:
    /// - [`AllocationProperty::File`]: Uses pinned memory staging buffers for faster
    ///   DMA transfers (useful for memory-mapped files)
    /// - [`AllocationProperty::Native`]: Data is in heap memory
    /// - [`AllocationProperty::Other`]: Unknown backing storage
    ///
    /// # Example
    ///
    /// ```
    /// use cubecl_common::bytes::{SharedBytesAllocationController, AllocationProperty};
    ///
    /// // From static data (zero-copy, no heap allocation)
    /// let static_bytes = bytes::Bytes::from_static(&[1, 2, 3, 4]);
    /// let controller = SharedBytesAllocationController::new(static_bytes, AllocationProperty::Other);
    ///
    /// // Memory-mapped file data - use File property for optimized GPU transfers
    /// let mmap_bytes = bytes::Bytes::from_static(&[1, 2, 3, 4]); // pretend this is mmap
    /// let controller = SharedBytesAllocationController::new(mmap_bytes, AllocationProperty::File);
    /// ```
    pub fn new(bytes: bytes::Bytes, property: AllocationProperty) -> Self {
        Self {
            bytes,
            controller: UnsafeCell::new(None),
            init: AtomicBool::new(false),
            property,
        }
    }

    /// Copy the shared bytes into a mutable native allocation controller.
    /// This is called lazily on first mutable access (copy-on-write).
    ///
    /// The allocation uses `MAX_ALIGN` alignment to ensure `try_into_vec` works
    /// for all tensor element types (f16, f32, f64, etc.).
    fn init_mutable(&self) {
        if self.init.load(Ordering::Relaxed) {
            return;
        }

        let data: &[u8] = &self.bytes;

        // Allocate with MAX_ALIGN to support all tensor element types in try_into_vec.
        // This ensures alignment is sufficient for f64, u128, SIMD types, etc.
        let controller = NativeAllocationController::alloc_with_data(data, MAX_ALIGN)
            .unwrap_or_else(|e| {
                panic!(
                    "failed to allocate MAX_ALIGN buffer for copy-on-write (len: {}, error: {:?})",
                    data.len(),
                    e
                )
            });

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
        // Always report MAX_ALIGN because that's what try_detach will provide.
        // This allows try_into_vec to succeed for all tensor element types.
        //
        // Before mutation: the actual bytes::Bytes data may have lower alignment,
        // but try_detach will trigger init_mutable which allocates with MAX_ALIGN.
        //
        // After mutation: the NativeAllocationController has MAX_ALIGN.
        MAX_ALIGN
    }

    fn property(&self) -> AllocationProperty {
        self.property
    }

    fn memory(&self) -> &[MaybeUninit<u8>] {
        if self.init.load(Ordering::Relaxed) {
            // Data has been copied to mutable controller, use that.
            // SAFETY: init is true, so controller is guaranteed to be Some.
            unsafe {
                (*self.controller.get())
                    .as_ref()
                    .expect("controller must be Some when init is true")
                    .memory()
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

        // SAFETY: init_mutable() guarantees the controller is Some.
        unsafe {
            (*self.controller.get())
                .as_mut()
                .expect("controller must be Some after init_mutable()")
                .memory_mut()
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
            Box::new(SharedBytesAllocationController::new(left, self.property)),
            Box::new(SharedBytesAllocationController::new(right, self.property)),
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
            self.property,
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
        // Trigger copy-on-write if not already done.
        // This copies the data into a NativeAllocationController with MAX_ALIGN alignment,
        // which can then be detached and converted to a Vec.
        self.init_mutable();

        // Now we have a NativeAllocationController that CAN be detached.
        // SAFETY: init_mutable guarantees controller is Some.
        unsafe {
            (*self.controller.get())
                .as_mut()
                .expect("controller must be Some after init_mutable()")
                .try_detach()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Bytes;
    use super::*;

    #[test_log::test]
    fn test_from_static() {
        static DATA: &[u8] = &[1, 2, 3, 4, 5];
        let shared = bytes::Bytes::from_static(DATA);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        assert_eq!(&bytes[..], &[1, 2, 3, 4, 5]);
        assert_eq!(bytes.len(), 5);
    }

    #[test_log::test]
    fn test_from_vec() {
        let shared = bytes::Bytes::from(alloc::vec![10, 20, 30]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Native);

        assert_eq!(&bytes[..], &[10, 20, 30]);
        assert_eq!(bytes.len(), 3);
    }

    #[test_log::test]
    fn test_split() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4, 5, 6]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        let (left, right) = bytes.split(3).unwrap();

        assert_eq!(&left[..], &[1, 2, 3]);
        assert_eq!(&right[..], &[4, 5, 6]);
    }

    #[test_log::test]
    fn test_split_at_zero() {
        // Boundary case: split at 0 creates empty left, full right
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        let (left, right) = bytes.split(0).unwrap();

        assert_eq!(left.len(), 0);
        assert_eq!(&right[..], &[1, 2, 3, 4]);
    }

    #[test_log::test]
    fn test_split_at_len() {
        // Boundary case: split at len creates full left, empty right
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);
        let len = bytes.len();

        let (left, right) = bytes.split(len).unwrap();

        assert_eq!(&left[..], &[1, 2, 3, 4]);
        assert_eq!(right.len(), 0);
    }

    #[test_log::test]
    fn test_duplicate() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = SharedBytesAllocationController::new(shared, AllocationProperty::Other);

        let dup = controller.duplicate().expect("duplicate should succeed");
        assert_eq!(dup.memory().len(), 3);
    }

    #[test_log::test]
    fn test_copy_into() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4, 5]);
        let controller = SharedBytesAllocationController::new(shared, AllocationProperty::Other);

        let mut buf = [0u8; 3];
        unsafe { controller.copy_into(&mut buf) };
        assert_eq!(buf, [1, 2, 3]);
    }

    #[test_log::test]
    fn test_property_file() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = SharedBytesAllocationController::new(shared, AllocationProperty::File);

        assert!(matches!(controller.property(), AllocationProperty::File));
    }

    #[test_log::test]
    fn test_bytes_from_shared_with_file_property() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::File);

        assert!(matches!(bytes.property(), AllocationProperty::File));
        assert_eq!(&bytes[..], &[1, 2, 3, 4]);
    }

    #[test_log::test]
    fn test_split_preserves_property() {
        // Verify that split preserves the allocation property
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4, 5, 6]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::File);

        let (left, right) = bytes.split(3).unwrap();

        assert!(matches!(left.property(), AllocationProperty::File));
        assert!(matches!(right.property(), AllocationProperty::File));
    }

    #[test_log::test]
    fn test_duplicate_preserves_property() {
        // Verify that duplicate preserves the allocation property
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::File);

        let cloned = bytes.clone();

        assert!(matches!(cloned.property(), AllocationProperty::File));
    }

    #[test_log::test]
    fn test_alignment_reports_max_align() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = SharedBytesAllocationController::new(shared, AllocationProperty::Other);

        // Always reports MAX_ALIGN because that's what try_detach will provide
        assert_eq!(controller.alloc_align(), MAX_ALIGN);
    }

    #[test_log::test]
    fn test_grow_fails() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let mut controller =
            SharedBytesAllocationController::new(shared, AllocationProperty::Other);

        let result = controller.grow(100, 1);
        assert!(matches!(result, Err(AllocationError::UnsupportedOperation)));
    }

    #[test_log::test]
    fn test_try_detach_always_succeeds() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4]);
        let mut controller =
            SharedBytesAllocationController::new(shared, AllocationProperty::Other);

        // try_detach triggers init_mutable internally and always succeeds.
        // This enables try_into_vec to work for SharedBytesAllocationController.
        let ptr = controller.try_detach();
        assert!(ptr.is_some(), "try_detach should always succeed");

        // Clean up: the detached memory needs to be deallocated
        if let Some(ptr) = ptr {
            unsafe {
                // Capacity is rounded up to MAX_ALIGN boundary
                let capacity = 4usize.next_multiple_of(MAX_ALIGN);
                let layout = core::alloc::Layout::from_size_align(capacity, MAX_ALIGN)
                    .expect("valid layout");
                alloc::alloc::dealloc(ptr.as_ptr(), layout);
            }
        }
    }

    #[test_log::test]
    fn test_try_into_vec_succeeds_for_u8() {
        // try_into_vec should work because try_detach triggers init_mutable
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        let result = bytes.try_into_vec::<u8>();
        assert!(
            result.is_ok(),
            "try_into_vec should succeed for shared bytes"
        );

        let vec = result.unwrap();
        assert_eq!(vec, alloc::vec![1, 2, 3, 4]);
    }

    #[test_log::test]
    fn test_try_into_vec_succeeds_for_f32() {
        // try_into_vec should work for f32 because alloc_align reports MAX_ALIGN
        // Use aligned static data - real tensor data from files is always aligned
        #[repr(align(4))]
        struct AlignedData([u8; 16]);

        static DATA: AlignedData = AlignedData({
            let f32_bytes: [[u8; 4]; 4] = [
                1.0f32.to_le_bytes(),
                2.0f32.to_le_bytes(),
                3.0f32.to_le_bytes(),
                4.0f32.to_le_bytes(),
            ];
            let mut result = [0u8; 16];
            let mut i = 0;
            while i < 4 {
                let mut j = 0;
                while j < 4 {
                    result[i * 4 + j] = f32_bytes[i][j];
                    j += 1;
                }
                i += 1;
            }
            result
        });
        let shared = bytes::Bytes::from_static(&DATA.0);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        let result = bytes.try_into_vec::<f32>();
        assert!(
            result.is_ok(),
            "try_into_vec::<f32> should succeed for shared bytes"
        );

        let vec = result.unwrap();
        assert_eq!(vec, alloc::vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test_log::test]
    fn test_try_into_vec_succeeds_for_f64() {
        // try_into_vec should work for f64 because alloc_align reports MAX_ALIGN
        // Use aligned static data - real tensor data from files is always aligned
        #[repr(align(16))]
        struct AlignedData([u8; 16]);

        static DATA: AlignedData = AlignedData({
            let f64_bytes: [[u8; 8]; 2] = [1.0f64.to_le_bytes(), 2.0f64.to_le_bytes()];
            let mut result = [0u8; 16];
            let mut i = 0;
            while i < 2 {
                let mut j = 0;
                while j < 8 {
                    result[i * 8 + j] = f64_bytes[i][j];
                    j += 1;
                }
                i += 1;
            }
            result
        });
        let shared = bytes::Bytes::from_static(&DATA.0);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        let result = bytes.try_into_vec::<f64>();
        assert!(
            result.is_ok(),
            "try_into_vec::<f64> should succeed for shared bytes"
        );

        let vec = result.unwrap();
        assert_eq!(vec, alloc::vec![1.0f64, 2.0]);
    }

    #[test_log::test]
    fn test_copy_on_write() {
        // Test that mutable access triggers copy-on-write
        let shared = bytes::Bytes::from_static(&[1, 2, 3, 4, 5]);
        let mut bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        // Mutate the first byte
        bytes[0] = 99;

        // Verify the mutation worked
        assert_eq!(bytes[0], 99);
        assert_eq!(&bytes[1..], &[2, 3, 4, 5]);
    }

    #[test_log::test]
    fn test_clone_before_mutation_is_cheap() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        // Clone should use duplicate() which is cheap (reference counted)
        let cloned = bytes.clone();

        assert_eq!(&bytes[..], &cloned[..]);
    }

    #[test_log::test]
    fn test_clone_after_mutation_copies() {
        let shared = bytes::Bytes::from_static(&[1, 2, 3]);
        let mut bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        // Trigger copy-on-write
        bytes[0] = 99;

        // Clone after mutation should do a full copy
        let cloned = bytes.clone();

        assert_eq!(&bytes[..], &cloned[..]);
        assert_eq!(cloned[0], 99);
    }

    #[test_log::test]
    fn test_slices_from_static_region() {
        // Simulate a static embedded data region (e.g., from include_bytes!)
        static EMBEDDED_DATA: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Create a shared bytes from the static region
        let shared = bytes::Bytes::from_static(EMBEDDED_DATA);

        // Slice into two parts: first 4 bytes and remaining 6 bytes
        let first_4 = shared.slice(0..4);
        let last_6 = shared.slice(4..10);

        // Create cubecl Bytes from each slice (zero-copy)
        let bytes_first = Bytes::from_shared(first_4, AllocationProperty::Other);
        let bytes_last = Bytes::from_shared(last_6, AllocationProperty::Other);

        // Verify contents
        assert_eq!(&bytes_first[..], &[1, 2, 3, 4]);
        assert_eq!(&bytes_last[..], &[5, 6, 7, 8, 9, 10]);

        // Verify lengths
        assert_eq!(bytes_first.len(), 4);
        assert_eq!(bytes_last.len(), 6);

        // Both should report MAX_ALIGN (what try_detach will provide)
        assert_eq!(bytes_first.align(), MAX_ALIGN);
        assert_eq!(bytes_last.align(), MAX_ALIGN);
    }

    #[test_log::test]
    fn test_multiple_slices_share_underlying_data() {
        // Demonstrate that multiple slices can be created from the same static region
        // without copying, and they all remain valid
        static DATA: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

        let shared = bytes::Bytes::from_static(DATA);

        // Create multiple overlapping and non-overlapping slices
        let slice_a = Bytes::from_shared(shared.slice(0..4), AllocationProperty::Other); // DEADBEEF
        let slice_b = Bytes::from_shared(shared.slice(4..8), AllocationProperty::Other); // CAFEBABE
        let slice_c = Bytes::from_shared(shared.slice(2..6), AllocationProperty::Other); // BEEF CAFE (overlapping)

        assert_eq!(&slice_a[..], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(&slice_b[..], &[0xCA, 0xFE, 0xBA, 0xBE]);
        assert_eq!(&slice_c[..], &[0xBE, 0xEF, 0xCA, 0xFE]);
    }
}
