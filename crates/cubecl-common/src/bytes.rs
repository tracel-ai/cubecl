//! A version of [`bytemuck::BoxBytes`] that is cloneable and allows trailing uninitialized elements.

use alloc::vec::Vec;
use core::alloc::{Layout, LayoutError};
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::bytes::default_allocation::DefaultAllocationController;

/// A sort of `Box<[u8]>` that remembers the original alignment and can contain trailing uninitialized bytes.
pub struct Bytes {
    data: Data,
    // SAFETY: The first `len` bytes of the allocation are initialized
    len: usize,
}

/// Internally used to avoid accidentally leaking an allocation or using the wrong layout.
struct Data {
    allocation: Allocation,
    controller: Box<dyn AllocationController>,
}

/// Contains the parts from an allocation.
pub struct Allocation {
    ptr: NonNull<u8>,
    size: usize,
    align: usize,
}

/// Defines how the current data is allocated and provides some operations on it.
pub trait AllocationController {
    /// Deallocates the provided [ptr](NonNull<u8>).
    fn dealloc(&self, allocation: &Allocation);
    /// Extend the current.
    fn grow(
        &self,
        allocation: &Allocation,
        size: usize,
        align: usize,
    ) -> Result<Allocation, AllocationError>;
    /// Returns wheter the [alloc crate](alloc) is used for allocation and if the allocation can be
    /// handled by another data structure.
    ///
    /// This means the allocation isn't behind a memory pool and can be safefy deallocated using
    /// the [alloc crate](alloc).
    ///
    /// # Notes
    ///
    /// This allows the ptr to be converted into a native Rust vector without new allocation.
    fn can_be_detached(&self) -> bool;
}

/// Error related to allocation.
#[derive(Debug, Clone)]
pub enum AllocationError {
    /// Can't grow the current allocation.
    CantGrow,
}

impl Bytes {
    /// Creates the type from its raw parts.
    pub unsafe fn from_raw_parts(
        ptr: NonNull<u8>,
        len: usize,
        align: usize,
        controller: Box<dyn AllocationController>,
    ) -> Self {
        let allocation = Allocation {
            ptr,
            size: len,
            align,
        };
        Self {
            data: Data {
                allocation,
                controller,
            },
            len,
        }
    }

    /// Create a sequence of [Bytes] from the memory representation of an unknown type of elements.
    /// Prefer this over [Self::from_elems] when the datatype is not statically known and erased at runtime.
    pub fn from_bytes_vec(bytes: Vec<u8>) -> Self {
        let mut bytes = Self::from_elems(bytes);
        // TODO: this method could be datatype aware and enforce a less strict alignment.
        // On most platforms, this alignment check is fulfilled either way though, so
        // the benefits of potentially saving a memcopy are negligible.
        bytes
            .try_enforce_runtime_align(default_allocation::MAX_ALIGN)
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
                core::mem::align_of::<E>() <= default_allocation::MAX_ALIGN,
                "element type not supported due to too large alignment"
            );
        };
        // Note: going through a Box as in Vec::into_boxed_slice would re-allocate on excess capacity. Avoid that.
        let byte_len = elems.len() * core::mem::size_of::<E>();
        let alloc = Data::from_vec(elems);
        Self {
            data: alloc,
            len: byte_len,
        }
    }

    /// Extend the byte buffer from a slice of bytes
    pub fn extend_from_byte_slice(&mut self, bytes: &[u8]) {
        self.extend_from_byte_slice_aligned(bytes, default_allocation::MAX_ALIGN)
    }

    /// Get the total capacity, in bytes, of the wrapped allocation.
    pub fn capacity(&self) -> usize {
        self.data.allocation.size
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
        let mut vec = match self.data.try_into_vec::<E>() {
            Ok(vec) => vec,
            Err(alloc) => {
                self.data = alloc;
                return Err(self);
            }
        };
        // SAFETY: We computed this length from the bytemuck-ed slice into this allocation
        unsafe {
            vec.set_len(length);
        };
        Ok(vec)
    }

    /// Get the alignment of the wrapped allocation.
    pub(crate) fn align(&self) -> usize {
        self.data.allocation.align
    }

    /// Extend the byte buffer from a slice of bytes.
    ///
    /// This is used internally to preserve the alignment of the memory layout when matching elements
    /// are extended. Prefer [`Self::extend_from_byte_slice`] otherwise.
    pub(crate) fn extend_from_byte_slice_aligned(&mut self, bytes: &[u8], align: usize) {
        let additional = bytes.len();
        self.reserve(additional, align);

        let len = self.len();
        let new_cap = len.wrapping_add(additional); // Can not overflow, as we've just successfully reserved sufficient space for it
        let uninit_spare = &mut self.data.memory_mut()[len..new_cap];
        // SAFETY: reinterpreting the slice as a MaybeUninit<u8>.
        // See also #![feature(maybe_uninit_write_slice)], which would replace this with safe code
        uninit_spare.copy_from_slice(unsafe {
            core::slice::from_raw_parts(bytes.as_ptr().cast(), additional)
        });
        self.len = new_cap;
    }

    /// Copy an existing slice of data into Bytes that are aligned to `align`
    fn try_from_data(align: usize, data: &[u8]) -> Result<Self, LayoutError> {
        let len = data.len();
        let layout = Layout::from_size_align(len, align)?;
        let alloc = Data::new(layout);
        unsafe {
            // SAFETY:
            // - data and alloc are distinct allocations of `len` bytes
            let data_ptr = data.as_ptr();
            core::ptr::copy_nonoverlapping::<u8>(data_ptr, alloc.as_mut_ptr(), len);
        };
        Ok(Self { data: alloc, len })
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
        let needs_to_grow = additional > self.capacity().wrapping_sub(self.len());
        if !needs_to_grow {
            return;
        }
        let Some(required_cap) = self.len().checked_add(additional) else {
            alloc_overflow()
        };
        // guarantee exponential growth for amortization
        let new_cap = required_cap.max(self.capacity() * 2);
        let new_cap = new_cap.max(align); // Small allocations would be pointless

        self.data.grow(new_cap, align)
    }
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: see type invariants
        unsafe { core::slice::from_raw_parts(self.data.as_mut_ptr(), self.len) }
    }
}

impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: see type invariants
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr(), self.len) }
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
        let align = default_allocation::MAX_ALIGN;
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

impl Data {
    // Wrap the allocation of a vector without copying
    fn from_vec<E: Copy>(vec: Vec<E>) -> Self {
        let mut elems = core::mem::ManuallyDrop::new(vec);
        // Set the length to 0, then all data is in the "spare capacity".
        // SAFETY: Data is Copy, so in particular does not need to be dropped. In any case, try not to panic until
        //  we have taken ownership of the data!
        unsafe { elems.set_len(0) };
        let data = elems.spare_capacity_mut();
        // We now have one contiguous slice of data to pass to Layout::for_value.
        let layout = Layout::for_value(data);
        // SAFETY: data is the allocation of a vec, hence can not be null. We use unchecked to avoid a panic-path.
        let ptr = unsafe { NonNull::new_unchecked(elems.as_mut_ptr().cast()) };
        let controller = DefaultAllocationController;
        let allocation = Allocation {
            ptr,
            size: layout.size(),
            align: layout.align(),
        };

        Self {
            allocation,
            controller: Box::new(controller),
        }
    }

    // Create a new allocation with the specified layout
    fn new(layout: Layout) -> Self {
        let ptr = default_allocation::buffer_alloc(layout);
        let allocation = Allocation {
            ptr,
            size: layout.size(),
            align: layout.align(),
        };
        let contoller = DefaultAllocationController;
        Self {
            allocation,
            controller: Box::new(contoller),
        }
    }

    // Returns a mutable view of the memory of the whole allocation
    fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: See type invariants
        unsafe {
            core::slice::from_raw_parts_mut(
                self.allocation.ptr.as_ptr().cast(),
                self.allocation.size,
            )
        }
    }

    fn grow(&mut self, size: usize, align: usize) {
        match self.controller.grow(&self.allocation, size, align) {
            Ok(allocation) => {
                self.allocation = allocation;
            }
            Err(_err) => {
                let mut new = Self::new(Layout::from_size_align(size, align).unwrap());
                let data = unsafe {
                    core::slice::from_raw_parts(
                        self.allocation.ptr.as_ptr().cast(),
                        self.allocation.size,
                    )
                };
                new.memory_mut().copy_from_slice(data);
                *self = new;
            }
        }
    }

    // Return a pointer to the underlying allocation. This pointer is valid for reads and writes until the allocation is dropped or reallocated.
    fn as_mut_ptr(&self) -> *mut u8 {
        self.allocation.ptr.as_ptr()
    }

    // Try to convert the allocation to a Vec. The Vec has a length of 0 when returned, but correct capacity and pointer!
    fn try_into_vec<E>(self) -> Result<Vec<E>, Self> {
        if !self.controller.can_be_detached() {
            return Err(self);
        }

        let byte_capacity = self.allocation.size;
        let Some(capacity) = byte_capacity.checked_div(size_of::<E>()) else {
            return Err(self);
        };
        if capacity * size_of::<E>() != byte_capacity {
            return Err(self);
        };
        if self.allocation.align < align_of::<E>() {
            return Err(self);
        }
        // Okay, let's commit
        let ptr = self.allocation.ptr.as_ptr().cast();
        core::mem::forget(self);
        // SAFETY:
        // - ptr was allocated by the global allocator as per type-invariant
        // - `E` has the same alignment as indicated by the stored layout.
        // - capacity * size_of::<E> == layout.size()
        // - 0 <= capacity
        // - no bytes are claimed to be initialized
        // - the layout represents a valid allocation, hence has allocation size less than isize::MAX
        Ok(unsafe { Vec::from_raw_parts(ptr, 0, capacity) })
    }
}

impl Drop for Data {
    fn drop(&mut self) {
        self.controller.dealloc(&self.allocation);
    }
}

fn expect_dangling(align: usize, buffer: NonNull<u8>) {
    debug_assert!(
        buffer.as_ptr().wrapping_sub(align).is_null(),
        "expected a nullptr for size 0"
    );
}

#[cold]
fn alloc_overflow() -> ! {
    panic!("Overflow, too many elements")
}

/// Module that defines an allocation based on the [alloc crate](alloc).
pub(crate) mod default_allocation {
    use super::*;
    use alloc::alloc::Layout;

    /// The maximum supported alignment. The limit exists to not have to store alignment when serializing. Instead,
    /// the bytes are always over-aligned when deserializing to MAX_ALIGN.
    pub const MAX_ALIGN: usize = core::mem::align_of::<u128>();

    /// Default allocation controller using the [alloc crate](alloc).
    ///
    /// SAFETY:
    ///  - If `layout.size() > 0`, `ptr` points to a valid allocation from the global allocator
    ///    of the specified layout. The first `len` bytes are initialized.
    ///  - If `layout.size() == 0`, `ptr` is aligned to `layout.align()` and `len` is 0.
    ///    `ptr` is further suitable to be used as the argument for `Vec::from_raw_parts` see [buffer alloc]
    ///    for more details.
    pub struct DefaultAllocationController;

    impl AllocationController for DefaultAllocationController {
        fn dealloc(&self, allocation: &Allocation) {
            let layout =
                unsafe { Layout::from_size_align_unchecked(allocation.size, allocation.align) };
            buffer_dealloc(layout, allocation.ptr);
        }

        fn grow(
            &self,
            allocation: &Allocation,
            size: usize,
            align: usize,
        ) -> Result<Allocation, AllocationError> {
            let Ok(new_layout) = Layout::from_size_align(size, align) else {
                return Err(AllocationError::CantGrow);
            };

            // Check done before.
            let old_layout = unsafe { Layout::from_size_align_unchecked(size, align) };
            let (layout, ptr) = buffer_grow(old_layout, allocation.ptr, new_layout);

            Ok(Allocation {
                ptr,
                size: layout.size(),
                align: layout.align(),
            })
        }

        fn can_be_detached(&self) -> bool {
            true
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
            let ptr = core::ptr::null_mut::<u8>().wrapping_add(layout.align());
            // SAFETY: layout.align() is never 0
            unsafe { NonNull::new_unchecked(ptr) }
        } else {
            // SAFETY: layout has non-zero size.
            let ptr = unsafe { alloc::alloc::alloc(layout) };
            NonNull::new(ptr).unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout))
        }
    }

    // Deallocate a buffer of a Vec
    fn buffer_dealloc(layout: Layout, buffer: NonNull<u8>) {
        if layout.size() != 0 {
            // SAFETY: buffer comes from a Vec or from [`buffer_alloc`/`buffer_grow`].
            // The layout is the same as per type-invariants
            unsafe {
                alloc::alloc::dealloc(buffer.as_ptr(), layout);
            }
        } else {
            // An empty Vec does not allocate, hence nothing to dealloc
            expect_dangling(layout.align(), buffer);
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
            let ptr =
                unsafe { alloc::alloc::realloc(buffer.as_ptr(), old_layout, new_layout.size()) };
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
                old_buffer = buffer;
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
                new_buffer.as_ptr(),
                old_layout.size(),
            );
        }
        buffer_dealloc(old_layout, old_buffer);
        (new_layout, new_buffer)
    }
}

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
