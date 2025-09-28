use core::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

/// Represents a single contiguous memory allocation.
///
/// The allocation can be manipulated using the [AllocationController],
/// though some operations, such as [grow](AllocationController::grow), may not be supported by all
/// implementations.
///
/// # Safety
///
/// Manipulating this data structure is highly unsafe and is intended for
/// [AllocationController] implementations rather than [Bytes] users.
/// Prefer using the safe [Bytes] API if you want to access data.
pub struct Allocation<'a> {
    /// Points to the beginning of the allocation. Must be valid for the lifetime of the allocation.
    ptr: NonNull<MaybeUninit<u8>>,
    /// The number of bytes allocated
    size: usize,
    /// The memory alignment of the allocation
    align: usize,
    /// The lifetime this allocation is valid for.
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> Allocation<'a> {
    /// Read only view of the underlying memory of this allocation. Not all bytes are initialized.
    pub fn memory(&self) -> &[MaybeUninit<u8>] {
        // SAFETY: See type invariants
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Read and mutate the underlying memory of this allocation. Not all bytes are initialized.
    pub fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: See type invariants
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// The pointer to the underlying memory of this allocation.
    pub fn ptr(&mut self) -> NonNull<MaybeUninit<u8>> {
        self.ptr
    }

    /// The alignment of the underlying memory of this allocation.
    pub fn align(&self) -> usize {
        self.align
    }

    /// Create a new allocation for the given pointer, size, and alignment.
    ///
    /// # Safety
    ///
    /// - Ptr must not be used elsewhere.
    /// - Ptr must be valid for at least the lifetime of `'a`
    /// - The allocation must have the alignment specified by `align`
    /// - The allocation must be at least `size` bytes long
    pub unsafe fn new_init(ptr: NonNull<u8>, size: usize, align: usize) -> Self {
        // SAFETY: Caller gaurantees invariants.
        unsafe {
            // Fine to cast to MaybeUninit, as a initialized u8 is ok. We can safely write to it
            // as the ptr is not used elsewhere.
            Self::new_uninit(ptr.cast(), size, align)
        }
    }

    /// Create a new allocation for the given pointer, size, and alignment.
    ///
    /// # Safety
    ///
    /// - Ptr must not be used elsewhere.
    /// - Ptr must be valid for at least the lifetime of `'a`
    /// - The allocation must have the alignment specified by `align`
    /// - The allocation must be at least `size` bytes long
    pub unsafe fn new_uninit(ptr: NonNull<MaybeUninit<u8>>, size: usize, align: usize) -> Self {
        Self {
            ptr,
            size,
            align,
            _lifetime: PhantomData,
        }
    }
}
