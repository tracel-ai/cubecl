//! A version of [`bytemuck::BoxBytes`] that is cloneable and allows trailing uninitialized elements.

use crate::bytes::default_controller::{self, NativeAllocationController};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::alloc::LayoutError;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

/// A buffer similar to `Box<[u8]>` that supports custom memory alignment and allows trailing uninitialized bytes.
///
/// `Bytes` is designed for efficient memory management in specialized contexts.
/// It may use non-standard allocators, such as the CUDA SDK allocator for pinned memory, or leverage memory pooling to reduce allocation overhead.
///
/// # Safety
///
/// The first `len` bytes of the allocation are guaranteed to be initialized. Accessing bytes beyond `len` is undefined behavior unless explicitly initialized.
pub struct Bytes {
    /// The buffer used to store data.
    controller: Box<dyn AllocationController>,
    /// The length of data actually used and initialized in the current buffer.
    len: usize,
}

/// Defines how an [Allocation] can be controlled.
///
/// This trait enables type erasure of the allocator after an [Allocation] is created, while still
/// providing methods to modify or manage an existing [Allocation].
pub trait AllocationController {
    /// The alignment this allocation was created with.
    fn alloc_align(&self) -> usize;

    /// Returns a mutable view of the memory of the whole allocation
    /// # Safety
    ///
    /// Must only write initialized data to the buffer.
    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>];

    /// Returns a view of the memory of the whole allocation
    fn memory(&self) -> &[MaybeUninit<u8>];

    /// Extends the provided [Allocation] to a new size with specified alignment.
    ///
    /// # Errors
    ///
    /// Returns an [AllocationError] if the extension fails (e.g., due to insufficient memory or
    /// unsupported operation by the allocator).
    #[allow(unused_variables)]
    fn grow(&mut self, size: usize, align: usize) -> Result<(), AllocationError> {
        Err(AllocationError::UnsupportedOperation)
    }

    /// Indicates whether the allocation uses the Rust [alloc](alloc) crate and can be safely
    /// managed by another data structure.
    ///
    /// If `true`, the allocation is not managed by a memory pool and can be safely deallocated
    /// using the [alloc](alloc) crate.
    ///
    /// # Notes
    ///
    /// This allows the allocation's pointer to be converted into a native Rust `Vec` without
    /// requiring a new allocation.
    ///
    /// Implementing this incorrectly is unsafe and may lead to undefined behavior.
    fn try_detach(&mut self) -> Option<NonNull<u8>> {
        None
    }
}

/// Errors that may occur during memory allocation operations.
///
/// This enum represents possible failure cases when manipulating an [Allocation] using an
/// [AllocationController].
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationError {
    /// The requested allocation operation is not supported by the allocator.
    ///
    /// This may occur, for example, when attempting to grow an allocation with an allocator that
    /// does not support resizing.
    UnsupportedOperation,

    /// The allocation failed due to insufficient memory.
    ///
    /// This typically indicates that the system or allocator could not provide the requested
    /// amount of memory.
    OutOfMemory,
}

impl Bytes {
    /// Creates the type from its raw parts.
    ///
    /// # Safety
    ///
    /// This function is highly unsafe, the provided length must be the actual number of bytes initialized in the
    /// AllocationController
    pub unsafe fn from_controller(controller: Box<dyn AllocationController>, len: usize) -> Self {
        debug_assert!(
            len <= controller.memory().len(),
            "len must not exceed controller memory size"
        );
        Self { controller, len }
    }

    /// Create a sequence of [Bytes] from the memory representation of an unknown type of elements.
    /// Prefer this over [Self::from_elems] when the datatype is not statically known and erased at runtime.
    pub fn from_bytes_vec(bytes: Vec<u8>) -> Self {
        let mut bytes = Self::from_elems(bytes);
        // TODO: this method could be datatype aware and enforce a less strict alignment.
        // On most platforms, this alignment check is fulfilled either way though, so
        // the benefits of potentially saving a memcopy are negligible.
        bytes
            .try_enforce_runtime_align(default_controller::MAX_ALIGN)
            .unwrap();
        bytes
    }

    /// Erase the element type of a vector by converting into a sequence of [Bytes].
    ///
    /// In case the element type is not statically known at runtime, prefer to use [Self::from_bytes_vec].
    pub fn from_elems<E>(elems: Vec<E>) -> Self
    where
        // NoUninit implies Copy
        E: bytemuck::NoUninit + Send + Sync,
    {
        let _: () = const {
            assert!(
                core::mem::align_of::<E>() <= default_controller::MAX_ALIGN,
                "element type not supported due to too large alignment"
            );
        };

        // Note: going through a Box as in Vec::into_boxed_slice would re-allocate on excess capacity. Avoid that.
        let byte_len = elems.len() * core::mem::size_of::<E>();
        let controller = NativeAllocationController::from_elems(elems);

        Self {
            controller: Box::new(controller),
            len: byte_len,
        }
    }

    /// Extend the byte buffer from a slice of bytes
    pub fn extend_from_byte_slice(&mut self, bytes: &[u8]) {
        self.extend_from_byte_slice_aligned(bytes, default_controller::MAX_ALIGN)
    }

    /// Get the total capacity, in bytes, of the wrapped allocation.
    pub fn capacity(&self) -> usize {
        self.controller.memory().len()
    }

    /// Convert the bytes back into a vector. This requires that the type has the same alignment as the element
    /// type this [Bytes] was initialized with.
    /// This only returns with Ok(_) if the conversion can be done without a memcopy
    pub fn try_into_vec<E: bytemuck::CheckedBitPattern + bytemuck::NoUninit>(
        mut self,
    ) -> Result<Vec<E>, Self> {
        // See if the length is compatible
        let Ok(data) = bytemuck::checked::try_cast_slice_mut::<_, E>(&mut self) else {
            return Err(self);
        };
        let length = data.len();
        // If so, try to convert the allocation to a vec
        let byte_capacity = self.controller.memory().len();

        let Some(capacity) = byte_capacity.checked_div(size_of::<E>()) else {
            return Err(self);
        };
        if capacity * size_of::<E>() != byte_capacity {
            return Err(self);
        };
        if self.controller.alloc_align() < align_of::<E>() {
            return Err(self);
        }

        let Some(ptr) = self.controller.try_detach() else {
            return Err(self);
        };

        // SAFETY:
        // - ptr was allocated by the global allocator as per type-invariant
        // - `E` has the same alignment as indicated by the stored layout.
        // - capacity * size_of::<E> == layout.size()
        // - 0 <= capacity
        // - no bytes are claimed to be initialized
        // - the layout represents a valid allocation, hence has allocation size less than isize::MAX
        // - We computed the length from the bytemuck-ed slice into this allocation
        let vec = unsafe { Vec::from_raw_parts(ptr.as_ptr().cast(), length, capacity) };
        Ok(vec)
    }

    /// Get the alignment of the wrapped allocation.
    pub fn align(&self) -> usize {
        self.controller.alloc_align()
    }

    /// Extend the byte buffer from a slice of bytes.
    ///
    /// This is used internally to preserve the alignment of the memory layout when matching elements
    /// are extended. Prefer [`Self::extend_from_byte_slice`] otherwise.
    pub fn extend_from_byte_slice_aligned(&mut self, bytes: &[u8], align: usize) {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        debug_assert!(
            align <= default_controller::MAX_ALIGN,
            "alignment exceeds maximum supported alignment"
        );

        let additional = bytes.len();
        self.reserve(additional, align);

        let len = self.len();
        let new_cap = len.wrapping_add(additional); // Can not overflow, as we've just successfully reserved sufficient space for it
        debug_assert!(
            new_cap <= self.capacity(),
            "new capacity must not exceed allocated capacity"
        );

        unsafe {
            // SAFETY: Will only write initialized memory to this ptr.
            let uninit_spare = &mut self.controller.memory_mut()[len..new_cap];
            // SAFETY: reinterpreting the slice as a MaybeUninit<u8>.
            // See also #![feature(maybe_uninit_write_slice)], which would replace this with safe code
            uninit_spare.copy_from_slice(core::slice::from_raw_parts(
                bytes.as_ptr().cast(),
                additional,
            ));
        };
        self.len = new_cap;
    }

    /// Copy an existing slice of data into Bytes that are aligned to `align`
    fn try_from_data(align: usize, data: &[u8]) -> Result<Self, LayoutError> {
        let controller = NativeAllocationController::alloc_with_data(data, align)?;

        Ok(Self {
            controller: Box::new(controller),
            len: data.len(),
        })
    }

    /// Ensure the contained buffer is aligned to `align` by possibly moving it to a new buffer.
    fn try_enforce_runtime_align(&mut self, align: usize) -> Result<(), LayoutError> {
        if self.as_mut_ptr().align_offset(align) == 0 {
            // data is already aligned correctly
            return Ok(());
        }
        *self = Self::try_from_data(align, self)?;
        Ok(())
    }

    fn reserve(&mut self, additional: usize, align: usize) {
        debug_assert!(
            align <= default_controller::MAX_ALIGN,
            "alignment exceeds maximum supported alignment"
        );

        let needs_to_grow = additional > self.capacity().wrapping_sub(self.len());
        if !needs_to_grow {
            return;
        }
        let Some(required_cap) = self.len().checked_add(additional) else {
            default_controller::alloc_overflow()
        };
        // guarantee exponential growth for amortization
        let new_cap = required_cap.max(self.capacity() * 2);
        let new_cap = new_cap.max(align); // Small allocations would be pointless

        match self.controller.grow(new_cap, align) {
            Ok(()) => {}
            Err(_err) => {
                let new_controller: Box<dyn AllocationController> = Box::new(
                    NativeAllocationController::alloc_with_capacity(new_cap, align).unwrap(),
                );
                let mut new_bytes = Self {
                    controller: new_controller,
                    len: self.len,
                };
                // Copy memory into new bytes.
                new_bytes.copy_from_slice(&*self);
                *self = new_bytes;
            }
        }
    }
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        let memory = &self.controller.memory()[0..self.len];
        // SAFETY: By construction, bytes up to len are initialized.
        unsafe { core::slice::from_raw_parts(memory.as_ptr().cast(), memory.len()) }
    }
}

impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = self.len;
        // SAFETY: We only expose this as initialized memory so cannot write uninitialized memory to this slice.
        let slice = unsafe { &mut self.controller.memory_mut() };
        // Get initialized part of this slice.
        let memory = &mut slice[0..len];
        // SAFETY: By construction, bytes up to len are initialized.
        unsafe { core::slice::from_raw_parts_mut(memory.as_mut_ptr().cast(), memory.len()) }
    }
}

// SAFETY: Bytes behaves like a Box<[u8]> and can contain only elements that are themselves Send
unsafe impl Send for Bytes {}
// SAFETY: Bytes behaves like a Box<[u8]> and can contain only elements that are themselves Sync
unsafe impl Sync for Bytes {}

fn debug_from_fn<F: Fn(&mut core::fmt::Formatter<'_>) -> core::fmt::Result>(
    f: F,
) -> impl core::fmt::Debug {
    // See also: std::fmt::from_fn
    struct FromFn<F>(F);
    impl<F> core::fmt::Debug for FromFn<F>
    where
        F: Fn(&mut core::fmt::Formatter<'_>) -> core::fmt::Result,
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            (self.0)(f)
        }
    }
    FromFn(f)
}

impl core::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let data = &**self;
        let fmt_data = move |f: &mut core::fmt::Formatter<'_>| {
            if data.len() > 3 {
                // There is a nightly API `debug_more_non_exhaustive` which has `finish_non_exhaustive`
                f.debug_list().entries(&data[0..3]).entry(&"...").finish()
            } else {
                f.debug_list().entries(data).finish()
            }
        };
        f.debug_struct("Bytes")
            .field("data", &debug_from_fn(fmt_data))
            .field("len", &self.len)
            .finish()
    }
}

impl serde::Serialize for Bytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_bytes::serialize(self.deref(), serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Bytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[cold]
        fn too_large<E: serde::de::Error>(len: usize, align: usize) -> E {
            // max_length = largest multiple of align that is <= isize::MAX
            // align is a power of 2, hence a multiple has the lower bits unset. Mask them off to find the largest multiple
            let max_length = (isize::MAX as usize) & !(align - 1);
            E::custom(core::format_args!(
                "length too large: {len}. Expected at most {max_length} bytes"
            ))
        }

        // TODO: we can possibly avoid one copy here by deserializing into an existing, correctly aligned, slice of bytes.
        // We might not be able to predict the length of the data, hence it's far more convenient to let `Vec` handle the growth and re-allocations.
        // Further, on a lot of systems, the allocator naturally aligns data to some reasonably large alignment, where no further copy is then
        // necessary.
        let data: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        // When deserializing, we over-align the data. This saves us from having to encode the alignment (which is platform-dependent in any case).
        // If we had more context information here, we could enforce some (smaller) alignment per data type. But this information is only available
        // in `TensorData`. Moreover it depends on the Deserializer there whether the datatype or data comes first.
        let align = default_controller::MAX_ALIGN;
        let mut bytes = Self::from_elems(data);
        bytes
            .try_enforce_runtime_align(align)
            .map_err(|_| too_large(bytes.len(), align))?;
        Ok(bytes)
    }
}

impl Clone for Bytes {
    fn clone(&self) -> Self {
        // unwrap here: the layout is valid as it has the alignment & size of self
        Self::try_from_data(self.align(), self.deref()).unwrap()
    }
}

impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for Bytes {}

#[cfg(test)]
mod tests {
    use super::Bytes;
    use alloc::{vec, vec::Vec};

    const _CONST_ASSERTS: fn() = || {
        fn test_send<T: Send>() {}
        fn test_sync<T: Sync>() {}
        test_send::<Bytes>();
        test_sync::<Bytes>();
    };

    fn test_serialization_roundtrip(bytes: &Bytes) {
        let config = bincode::config::standard();
        let serialized =
            bincode::serde::encode_to_vec(bytes, config).expect("serialization to succeed");
        let (roundtripped, _) = bincode::serde::decode_from_slice(&serialized, config)
            .expect("deserialization to succeed");
        assert_eq!(
            bytes, &roundtripped,
            "roundtripping through serialization didn't lead to equal Bytes"
        );
    }

    #[test]
    fn test_serialization() {
        test_serialization_roundtrip(&Bytes::from_elems::<i32>(vec![]));
        test_serialization_roundtrip(&Bytes::from_elems(vec![0xdead, 0xbeaf]));
    }

    #[test]
    fn test_into_vec() {
        // We test an edge case here, where the capacity (but not actual size) makes it impossible to convert to a vec
        let mut bytes = Vec::with_capacity(6);
        let actual_cap = bytes.capacity();
        bytes.extend_from_slice(&[0, 1, 2, 3]);
        let mut bytes = Bytes::from_elems::<u8>(bytes);

        bytes = bytes
            .try_into_vec::<[u8; 0]>()
            .expect_err("Conversion should not succeed for a zero-sized type");
        if actual_cap % 4 != 0 {
            // We most likely get actual_cap == 6, we can't force Vec to actually do that. Code coverage should complain if the actual test misses this
            bytes = bytes.try_into_vec::<[u8; 4]>().err().unwrap_or_else(|| {
                panic!("Conversion should not succeed due to capacity {actual_cap} not fitting a whole number of elements");
            });
        }
        bytes = bytes
            .try_into_vec::<u16>()
            .expect_err("Conversion should not succeed due to mismatched alignment");
        bytes = bytes.try_into_vec::<[u8; 3]>().expect_err(
            "Conversion should not succeed due to size not fitting a whole number of elements",
        );
        let bytes = bytes.try_into_vec::<[u8; 2]>().expect("Conversion should succeed for bit-convertible types of equal alignment and compatible size");
        assert_eq!(bytes, &[[0, 1], [2, 3]]);
    }

    #[test]
    fn test_grow() {
        let mut bytes = Bytes::from_elems::<u8>(vec![]);
        bytes.extend_from_byte_slice(&[0, 1, 2, 3]);
        assert_eq!(bytes[..], [0, 1, 2, 3][..]);

        let mut bytes = Bytes::from_elems(vec![42u8; 4]);
        bytes.extend_from_byte_slice(&[0, 1, 2, 3]);
        assert_eq!(bytes[..], [42, 42, 42, 42, 0, 1, 2, 3][..]);
    }

    #[test]
    fn test_large_elems() {
        let mut bytes = Bytes::from_elems(vec![42u128]);
        const TEST_BYTES: [u8; 16] = [
            0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, 0x56,
            0x34, 0x12,
        ];
        bytes.extend_from_byte_slice(&TEST_BYTES);
        let vec = bytes.try_into_vec::<u128>().unwrap();
        assert_eq!(vec, [42u128, u128::from_ne_bytes(TEST_BYTES)]);
    }
}
