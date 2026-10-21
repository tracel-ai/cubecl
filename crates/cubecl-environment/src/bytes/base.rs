//! A version of [`bytemuck::BoxBytes`] that is cloneable and allows trailing uninitialized elements.

use crate::bytes::{
    AccessError, AccessPolicy, BufferAllocator, Reader, Writer,
    allocator::AllocatorController,
    default_controller::{self, Global, NativeAllocationController},
    shared_arc::SharedAllocationController,
};
use crate::sync::Arc;
use alloc::{boxed::Box, vec::Vec};
use core::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The kind of allocation behind the [Bytes] type.
pub enum AllocationProperty {
    /// A file is used to store the data.
    File,
    /// The native allocator of Rust is used.
    Native,
    /// Pinned memory is used.
    Pinned,
    /// The data still lives on a compute device and is only copied to host memory
    /// lazily on first access (see [`Client::read_lazy`](https://docs.rs/cubecl-runtime)).
    Device,
    /// Another kind of memory is used.
    Other,
}

/// Error when splitting an allocation.
#[derive(Debug, Clone, Copy)]
pub enum SplitError {
    /// The offset isn't valid.
    InvalidOffset,
    /// The operation isn't supported.
    Unsupported,
}

/// Error when taking a [view](Bytes::view) into an allocation.
#[derive(Debug, Clone, Copy)]
pub enum ViewError {
    /// The range is out of bounds.
    InvalidRange,
    /// The backend can't produce a zero-copy window.
    Unsupported,
}

/// Controls how [`Bytes::split`] behaves when the allocation can't be split in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitPolicy {
    /// Prefer zero-copy splits, allowing the allocation to be shared behind an
    /// [`Arc`] when it can't be split natively.
    ///
    /// Mutating a resulting half then triggers a copy-on-write. This is the best
    /// choice for read-only or clone-heavy use cases, where the extra copy is
    /// never paid.
    Shared,
    /// Ensure both halves are independently owned, copying eagerly when a native
    /// split isn't possible.
    ///
    /// This avoids a later copy-on-write surprise when the halves will be
    /// mutated. Prefer it when you split *in order to* mutate, so the copy is
    /// paid once, up front, instead of lazily on first write.
    Owned,
}

/// Defines how an ``Allocation`` can be controlled.
///
/// This trait enables type erasure of the allocator after an ``Allocation`` is created, while still
/// providing methods to modify or manage an existing ``Allocation``.
pub trait AllocationController {
    /// The alignment this allocation was created with.
    fn alloc_align(&self) -> usize;

    /// Returns memory property for the current allocation.
    fn property(&self) -> AllocationProperty;

    /// Returns a mutable view of the memory of the whole allocation, subject to `policy`.
    ///
    /// Returns [`AccessError::WouldCopy`] if satisfying the access would require a copy
    /// (e.g. copy-on-write of shared storage) that `policy` forbids, or [`AccessError::Read`]
    /// if materializing lazy storage fails.
    ///
    /// # Safety
    ///
    /// Must only write initialized data to the buffer.
    unsafe fn memory_mut(
        &mut self,
        policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError>;

    /// Returns a view of the memory of the whole allocation, subject to `policy`.
    ///
    /// Returns [`AccessError::WouldCopy`] if satisfying the access would require a copy
    /// (e.g. materializing lazy storage) that `policy` forbids, or [`AccessError::Read`] if
    /// materializing fails.
    fn memory(&self, policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError>;

    /// The total byte capacity of the allocation, WITHOUT materializing lazy storage.
    ///
    /// The default is correct for already-resident controllers (where `memory()` is free);
    /// lazy controllers must override it to report their size cheaply so that querying the
    /// capacity never triggers a copy.
    fn capacity(&self) -> usize {
        match self.memory(AccessPolicy::allow_copy()) {
            Ok(memory) => memory.len(),
            // A resident controller using this default must never error here; a lazy
            // controller is expected to override `capacity` and not reach this path.
            Err(err) => {
                debug_assert!(false, "capacity(): resident host access failed: {err:?}");
                0
            }
        }
    }

    /// Splits the current allocation in multiple separate allocations.
    #[allow(clippy::type_complexity)]
    fn split(
        &mut self,
        _offset: usize,
    ) -> Result<(Box<dyn AllocationController>, Box<dyn AllocationController>), SplitError> {
        Err(SplitError::Unsupported)
    }

    /// Returns a zero-copy controller over the sub-range `[start, end)` of this
    /// allocation when the backend supports cheap sub-views (files and shared
    /// buffers), borrowing `&self`. Returns `None` otherwise.
    fn view(&self, _start: usize, _end: usize) -> Option<Box<dyn AllocationController>> {
        None
    }

    /// Duplicates the current allocation with a clone on write strategy if the allocation
    /// controller supports it.
    fn duplicate(&self) -> Option<Box<dyn AllocationController>> {
        None
    }

    /// Reads the data from the current allocation controller and copy its content into the provided
    /// buffer.
    ///
    /// # Safety
    ///
    /// Ensures the length provided reflect initialized values in the current allocation controller.
    unsafe fn copy_into(&self, buf: &mut [u8]) {
        let len = buf.len();
        let memory = self
            .memory(AccessPolicy::default())
            .expect("copy_into: host access failed");
        let memory_slice = &memory[0..len];

        // SAFETY: By construction, bytes up to len are initialized.
        let data = unsafe {
            core::slice::from_raw_parts(memory_slice.as_ptr().cast(), memory_slice.len())
        };
        buf.copy_from_slice(data);
    }

    /// Extends the provided ``Allocation`` to a new size with specified alignment.
    ///
    /// # Errors
    ///
    /// Returns an [`AllocationError`] if the extension fails (e.g., due to insufficient memory or
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

    /// Allocates `capacity` uninitialized bytes in the same memory as this allocation.
    ///
    /// [`Bytes`] calls this method when it copies its data to a new allocation, for example in
    /// `clone`, in `split` and in copy-on-write. Then the copy stays in the same memory as the
    /// original. `capacity` is a multiple of `align`.
    ///
    /// The default returns `None`. Then the new allocation comes from the global allocator.
    ///
    /// # Errors
    ///
    /// Returns `Some(Err(_))` when the memory of this allocation has no space for `capacity`
    /// bytes.
    #[allow(unused_variables)]
    fn alloc_like(
        &self,
        capacity: usize,
        align: usize,
    ) -> Option<Result<Box<dyn AllocationController>, AllocationError>> {
        None
    }
}

/// Allocates `capacity` uninitialized bytes in the same memory as `source`.
///
/// The capacity becomes a multiple of `align`. When `source` has no memory of its own, the
/// allocation comes from the global allocator.
pub(crate) fn alloc_like(
    source: &dyn AllocationController,
    capacity: usize,
    align: usize,
) -> Result<Box<dyn AllocationController>, AllocationError> {
    let capacity = capacity.next_multiple_of(align);
    match source.alloc_like(capacity, align) {
        Some(controller) => controller,
        None => NativeAllocationController::alloc_with_capacity(Global, capacity, align)
            .map(|controller| Box::new(controller) as Box<dyn AllocationController>),
    }
}

/// Copies `data` into a new allocation in the same memory as `source`.
pub(crate) fn copy_like(
    source: &dyn AllocationController,
    data: &[u8],
    align: usize,
) -> Result<Box<dyn AllocationController>, AllocationError> {
    let mut controller = alloc_like(source, data.len(), align)?;
    // SAFETY: Only initialized bytes are written.
    let memory = unsafe { controller.memory_mut(AccessPolicy::default()) }
        .expect("a new allocation is always host-resident");
    write_init(&mut memory[..data.len()], data);
    Ok(controller)
}

/// Writes `data` into `memory`. The two slices must have the same length.
fn write_init(memory: &mut [MaybeUninit<u8>], data: &[u8]) {
    // SAFETY: `&[u8]` and `&[MaybeUninit<u8>]` have the same layout, and the source is only read.
    // See also #![feature(maybe_uninit_write_slice)], which would replace this with safe code.
    memory
        .copy_from_slice(unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), data.len()) });
}

/// Errors that may occur during memory allocation operations.
///
/// This enum represents possible failure cases when manipulating an ``Allocation`` using an
/// [`AllocationController`].
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

impl core::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedOperation => {
                f.write_str("the allocator does not support this operation")
            }
            Self::OutOfMemory => f.write_str("the allocator has no memory for this allocation"),
        }
    }
}

impl core::error::Error for AllocationError {}

impl Bytes {
    /// Splits the current allocation at the given offset.
    ///
    /// Controllers that support a native split (such as files and shared
    /// buffers) always split in place. When they can't, the behaviour of the
    /// fallback is selected by `policy`:
    ///
    /// - [`SplitPolicy::Shared`] shares the allocation behind an [`Arc`] (via
    ///   [`Self::shared`]) so both halves reference the same backing memory
    ///   without copying. This is what makes [`Self::split`] work for native
    ///   allocations whose element alignment is larger than one byte. Mutating a
    ///   half then triggers a copy-on-write.
    /// - [`SplitPolicy::Owned`] copies eagerly so each half is independently
    ///   owned and can be mutated without a later copy-on-write.
    ///
    /// In both policies, when the allocation can be detached into a `Vec`
    /// without copying, that zero-copy path is taken first.
    pub fn split(
        self,
        offset: usize,
        policy: SplitPolicy,
    ) -> Result<(Bytes, Bytes), (Bytes, SplitError)> {
        if offset > self.len {
            return Err((self, SplitError::InvalidOffset));
        }
        match policy {
            SplitPolicy::Shared => self.split_shared(offset),
            SplitPolicy::Owned => self.split_owned(offset),
        }
    }

    /// Split preferring zero-copy: native split, else detach into a `Vec`, else
    /// share behind an [`Arc`]. See [`SplitPolicy::Shared`].
    fn split_shared(mut self, offset: usize) -> Result<(Bytes, Bytes), (Bytes, SplitError)> {
        let right_len = self.len - offset;
        match self.controller.split(offset) {
            Ok((left, right)) => unsafe {
                Ok((
                    Bytes::from_controller(left, offset),
                    Bytes::from_controller(right, right_len),
                ))
            },
            Err(_) => match self.try_into_vec::<u8>() {
                Ok(mut left) => {
                    let right = left.split_off(offset);

                    Ok((Bytes::from_bytes_vec(left), Bytes::from_bytes_vec(right)))
                }
                // The allocation can't be detached into a `Vec` (e.g. a native
                // allocation with element alignment > 1, or already shared
                // data). Share it behind an `Arc` so both halves reference the
                // same backing memory without copying.
                Err(this) => Ok(this.shared_split(offset)),
            },
        }
    }

    /// Split ensuring both halves are independently owned, copying eagerly when
    /// a zero-copy detach isn't possible. See [`SplitPolicy::Owned`].
    fn split_owned(self, offset: usize) -> Result<(Bytes, Bytes), (Bytes, SplitError)> {
        match self.try_into_vec::<u8>() {
            Ok(mut left) => {
                let right = left.split_off(offset);

                Ok((Bytes::from_bytes_vec(left), Bytes::from_bytes_vec(right)))
            }
            // The allocation can't be detached into a `Vec`. Copy each half into
            // its own allocation, in the same memory, so they can be mutated
            // without a later copy-on-write.
            Err(this) => {
                match (
                    this.copy_like(&this[..offset]),
                    this.copy_like(&this[offset..]),
                ) {
                    (Ok(left), Ok(right)) => Ok((left, right)),
                    // The memory of this allocation has no space for the copies.
                    _ => Err((this, SplitError::Unsupported)),
                }
            }
        }
    }

    /// Returns the sub-range `[start, end)` as a zero-copy [`Bytes`] window.
    ///
    /// The returned [`Bytes`] is independently owned (it shares the backing
    /// storage by reference count / re-opened file handle rather than borrowing
    /// `self`), so it can outlive this reference.
    ///
    /// Returns [`ViewError::Unsupported`] when the backend can't produce a
    /// zero-copy window (e.g. a plain heap allocation); [`Self::shared`] it first
    /// to make views available.
    pub fn view(&self, start: usize, end: usize) -> Result<Bytes, ViewError> {
        if start > end || end > self.len {
            return Err(ViewError::InvalidRange);
        }
        let len = end - start;

        match self.controller.view(start, end) {
            // SAFETY: the sub-view controller reports exactly `len` bytes.
            Some(controller) => Ok(unsafe { Bytes::from_controller(controller, len) }),
            None => Err(ViewError::Unsupported),
        }
    }

    /// Shares the current allocation behind an [`Arc`], returning a new [`Bytes`]
    /// that references the same data without copying.
    ///
    /// Cloning and [splitting](Self::split) the returned [`Bytes`] is cheap
    /// (reference counted, zero-copy). Because the data is shared,
    /// [`Self::try_into_vec`] will never succeed on the result, and mutating it
    /// triggers a copy-on-write into a private buffer.
    pub fn shared(self) -> Self {
        let len = self.len;
        let controller = SharedAllocationController::new(Arc::new(self), 0, len);

        Self {
            controller: Box::new(controller),
            len,
        }
    }

    /// Shares the allocation behind an [`Arc`] and returns two zero-copy views
    /// split at `offset`. The caller must ensure `offset <= self.len`.
    fn shared_split(self, offset: usize) -> (Bytes, Bytes) {
        let len = self.len;
        let inner = Arc::new(self);
        let left = SharedAllocationController::new(inner.clone(), 0, offset);
        let right = SharedAllocationController::new(inner, offset, len - offset);

        // SAFETY: each view reports exactly its own length as initialized memory.
        unsafe {
            (
                Bytes::from_controller(Box::new(left), offset),
                Bytes::from_controller(Box::new(right), len - offset),
            )
        }
    }

    #[cfg(feature = "std")]
    /// Creates bytes from a file at the given offset of the given size.
    pub fn from_file<P: Into<std::path::PathBuf>>(file: P, size: u64, offset: u64) -> Self {
        let controller = crate::bytes::file::FileAllocationController::new(file, size, offset);

        Self {
            controller: Box::new(controller),
            len: size as usize,
        }
    }

    /// Creates bytes from a shared [`bytes::Bytes`] buffer (zero-copy).
    ///
    /// This is useful for zero-copy tensor loading from:
    /// - Static embedded data via [`bytes::Bytes::from_static`]
    /// - Memory-mapped files
    /// - Any other [`bytes::Bytes`] source
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
    /// use cubecl_environment::bytes::{Bytes, AllocationProperty};
    ///
    /// // Memory-mapped file data - use File property for optimized GPU transfers
    /// let mmap_bytes = bytes::Bytes::from_static(&[1, 2, 3, 4]); // pretend this is mmap
    /// let bytes = Bytes::from_shared(mmap_bytes, AllocationProperty::File);
    /// assert!(matches!(bytes.property(), AllocationProperty::File));
    /// ```
    #[cfg(feature = "shared-bytes")]
    pub fn from_shared(bytes: bytes::Bytes, property: AllocationProperty) -> Self {
        let len = bytes.len();
        let controller =
            crate::bytes::shared::SharedBytesAllocationController::new(bytes, property);

        Self {
            controller: Box::new(controller),
            len,
        }
    }

    /// Creates bytes from a shared [`bytes::Bytes`] buffer (zero-copy), with copy-on-write into
    /// `allocator`.
    ///
    /// Reads use `bytes` directly, the same as [`Self::from_shared`]. The first write copies the
    /// data into memory from `allocator`, not into the global allocator.
    #[cfg(feature = "shared-bytes")]
    pub fn from_shared_in(
        bytes: bytes::Bytes,
        property: AllocationProperty,
        allocator: &'static dyn BufferAllocator,
    ) -> Self {
        let len = bytes.len();
        let controller = crate::bytes::shared::SharedBytesAllocationController::new_in(
            bytes, property, allocator,
        );

        Self {
            controller: Box::new(controller),
            len,
        }
    }

    /// Creates empty bytes with `capacity` bytes of memory from `allocator`.
    ///
    /// Clones, splits and growth of the result also get their memory from `allocator`. The
    /// memory never becomes a `Vec`: [`Self::try_into_vec`] always returns an error.
    ///
    /// # Errors
    ///
    /// - [`AllocationError::OutOfMemory`] when `allocator` has no memory for `capacity` bytes.
    /// - [`AllocationError::UnsupportedOperation`] when `align` is larger than 16 bytes.
    pub fn with_capacity_in(
        allocator: &'static dyn BufferAllocator,
        capacity: usize,
        align: usize,
    ) -> Result<Self, AllocationError> {
        let controller = AllocatorController::alloc_with_capacity(
            allocator,
            capacity.next_multiple_of(align),
            align,
        )?;

        Ok(Self {
            controller: Box::new(controller),
            len: 0,
        })
    }

    /// Creates `len` zero bytes in memory from `allocator`.
    ///
    /// See [`Self::with_capacity_in`] for the behaviour of the memory.
    ///
    /// # Errors
    ///
    /// The same as [`Self::with_capacity_in`].
    pub fn zeroed_in(
        allocator: &'static dyn BufferAllocator,
        len: usize,
        align: usize,
    ) -> Result<Self, AllocationError> {
        let mut bytes = Self::with_capacity_in(allocator, len, align)?;
        // SAFETY: Only initialized bytes are written.
        let memory = unsafe { bytes.memory_mut_default() };
        memory[..len].fill(MaybeUninit::new(0));
        bytes.len = len;
        Ok(bytes)
    }

    /// Creates a copy of `data` in memory from `allocator`.
    ///
    /// See [`Self::with_capacity_in`] for the behaviour of the memory.
    ///
    /// # Errors
    ///
    /// The same as [`Self::with_capacity_in`].
    pub fn from_slice_in(
        allocator: &'static dyn BufferAllocator,
        data: &[u8],
        align: usize,
    ) -> Result<Self, AllocationError> {
        let mut bytes = Self::with_capacity_in(allocator, data.len(), align)?;
        // SAFETY: Only initialized bytes are written.
        let memory = unsafe { bytes.memory_mut_default() };
        write_init(&mut memory[..data.len()], data);
        bytes.len = data.len();
        Ok(bytes)
    }

    /// Creates empty bytes with `capacity` bytes of memory from the global allocator.
    ///
    /// Write the data into [`Self::spare_capacity_mut`], then call [`Self::set_len`]. Unlike
    /// [`Self::with_capacity_in`], [`Self::try_into_vec`] can return this memory as a `Vec`
    /// without a copy, if `align` is the alignment of the element type.
    ///
    /// # Errors
    ///
    /// - [`AllocationError::UnsupportedOperation`] when `align` is larger than 16 bytes.
    /// - [`AllocationError::OutOfMemory`] when `capacity` is too large for a layout.
    ///
    /// When the global allocator has no memory, the program aborts, as with `Vec`.
    pub fn with_capacity(capacity: usize, align: usize) -> Result<Self, AllocationError> {
        let controller = NativeAllocationController::alloc_with_capacity(
            Global,
            capacity.next_multiple_of(align),
            align,
        )?;

        Ok(Self {
            controller: Box::new(controller),
            len: 0,
        })
    }

    /// Creates bytes that hold `f(0)`, `f(1)`, ..., `f(n - 1)`, in memory from `allocator`.
    ///
    /// The elements are written once, without zeroing the memory first. The alignment is the
    /// alignment of `E`. See [`Self::with_capacity_in`] for the behaviour of the memory.
    ///
    /// # Errors
    ///
    /// The same as [`Self::with_capacity_in`]. A size that overflows `usize` is
    /// [`AllocationError::OutOfMemory`].
    pub fn from_fn_in<E, F>(
        allocator: &'static dyn BufferAllocator,
        n: usize,
        f: F,
    ) -> Result<Self, AllocationError>
    where
        E: bytemuck::NoUninit,
        F: FnMut(usize) -> E,
    {
        let size = n
            .checked_mul(size_of::<E>())
            .ok_or(AllocationError::OutOfMemory)?;
        let mut bytes = Self::with_capacity_in(allocator, size, align_of::<E>())?;
        bytes.write_from_fn(n, f);
        Ok(bytes)
    }

    /// Creates bytes that hold `f(0)`, `f(1)`, ..., `f(n - 1)`, in memory from the global
    /// allocator.
    ///
    /// The result is the same as [`Self::from_elems`] of the collected elements, without the
    /// intermediate `Vec`.
    ///
    /// # Panics
    ///
    /// When the size overflows `usize`, or when the alignment of `E` is larger than 16 bytes.
    pub fn from_fn<E, F>(n: usize, f: F) -> Self
    where
        E: bytemuck::NoUninit,
        F: FnMut(usize) -> E,
    {
        let size = n.checked_mul(size_of::<E>()).expect("size overflows usize");
        let mut bytes = Self::with_capacity(size, align_of::<E>())
            .unwrap_or_else(|err| panic!("failed to allocate {size} bytes: {err}"));
        bytes.write_from_fn(n, f);
        bytes
    }

    /// Writes `f(0)`, ..., `f(n - 1)` after `len`, and adds them to `len`.
    ///
    /// The spare capacity must hold `n` elements, and `len` must be zero, so that the elements
    /// are aligned.
    fn write_from_fn<E, F>(&mut self, n: usize, mut f: F)
    where
        E: bytemuck::NoUninit,
        F: FnMut(usize) -> E,
    {
        debug_assert_eq!(self.len, 0, "the elements must start at an aligned offset");
        let spare = self.spare_capacity_mut();
        assert!(
            spare.len() >= n * size_of::<E>(),
            "the spare capacity is too small"
        );
        let ptr = spare.as_mut_ptr().cast::<E>();
        debug_assert!(
            ptr.is_aligned(),
            "the memory is not aligned for the element type"
        );
        for i in 0..n {
            // SAFETY: The spare capacity holds `n` elements, and the pointer is aligned to `E`.
            // When `f` panics, `len` does not change, and `E` is `Copy`, so nothing leaks.
            unsafe { ptr.add(i).write(f(i)) };
        }
        // SAFETY: The first `n * size_of::<E>()` bytes are now initialized.
        unsafe { self.set_len(n * size_of::<E>()) };
    }

    /// The uninitialized memory after the data: the bytes from `len` to the capacity.
    ///
    /// Write data here, then call [`Self::set_len`] to make it part of the bytes. If the memory
    /// is still shared, this copies it first (copy-on-write).
    ///
    /// # Panics
    ///
    /// When the memory is not accessible from the host.
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let len = self.len;
        // SAFETY: The bytes after `len` are not part of the initialized data, so writes of
        // uninitialized values there do not break the invariant of `Bytes`.
        let memory = unsafe { self.memory_mut_default() };
        &mut memory[len..]
    }

    /// Sets the length of the data.
    ///
    /// # Safety
    ///
    /// - `len` must not be larger than [`Self::capacity`].
    /// - The first `len` bytes must be initialized, for example by writes to
    ///   [`Self::spare_capacity_mut`].
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(
            len <= self.capacity(),
            "the length must not be larger than the capacity"
        );
        self.len = len;
    }

    /// The controller of the allocation.
    pub(crate) fn controller(&self) -> &dyn AllocationController {
        &*self.controller
    }

    /// Copies `data` into new bytes in the same memory as `self`, with the same alignment.
    fn copy_like(&self, data: &[u8]) -> Result<Self, AllocationError> {
        Ok(Self {
            controller: copy_like(&*self.controller, data, self.align())?,
            len: data.len(),
        })
    }

    /// The size of the allocation.
    ///
    /// # Notes
    ///
    /// This is used so that calling `bytes.len()` doesn't trigger [Deref], which may be expensive.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Read the bytes, configured by `reader`. Returns the initialized `[0, len)` slice.
    ///
    /// Unlike [`Deref`], this is fallible: it surfaces an [`AccessError`] instead of
    /// panicking, and a [`Reader::no_copy`] reader returns [`AccessError::WouldCopy`] rather
    /// than silently materializing lazy storage.
    pub fn read(&self, reader: Reader) -> Result<&[u8], AccessError> {
        let memory = &self.controller.memory(reader.policy)?[0..self.len];
        // SAFETY: bytes up to `len` are initialized by construction.
        Ok(unsafe { core::slice::from_raw_parts(memory.as_ptr().cast(), memory.len()) })
    }

    /// Mutably access the bytes, configured by `writer`. Returns the initialized `[0, len)`
    /// slice. A [`Writer::no_copy`] writer fails with [`AccessError::WouldCopy`] on
    /// still-shared buffers instead of triggering copy-on-write.
    pub fn write(&mut self, writer: Writer) -> Result<&mut [u8], AccessError> {
        let len = self.len;
        // SAFETY: we only ever expose initialized memory in `[0, len)`.
        let memory = unsafe { self.controller.memory_mut(writer.policy) }?;
        let memory = &mut memory[0..len];
        // SAFETY: bytes up to `len` are initialized by construction.
        Ok(unsafe { core::slice::from_raw_parts_mut(memory.as_mut_ptr().cast(), memory.len()) })
    }

    /// Default host access (allow copy), panicking on failure — backs [`Deref`].
    fn memory_default(&self) -> &[MaybeUninit<u8>] {
        self.controller
            .memory(AccessPolicy::default())
            .expect("bytes: host access failed")
    }

    /// Default mutable host access (allow copy-on-write), panicking on failure — backs
    /// [`DerefMut`].
    ///
    /// # Safety
    ///
    /// Caller must only write initialized data into the returned slice.
    unsafe fn memory_mut_default(&mut self) -> &mut [MaybeUninit<u8>] {
        unsafe { self.controller.memory_mut(AccessPolicy::default()) }
            .expect("bytes: host access failed")
    }

    /// Copy the data from the current allocation to the provided [Bytes].
    pub fn copy_into(&self, other: &mut Self) {
        unsafe {
            self.controller.copy_into(other);
        }
    }

    /// Retrieves the allocation property of the given allocation.
    pub fn property(&self) -> AllocationProperty {
        self.controller.property()
    }
    /// Creates the type from its raw parts.
    ///
    /// # Safety
    ///
    /// This function is highly unsafe, the provided length must be the actual number of bytes
    /// initialized in the `AllocationController`.
    ///
    /// Note: we intentionally do not assert `len <= controller.memory().len()` here, as
    /// `memory()` may force a lazy controller to materialize its data (e.g. a device-backed
    /// allocation), defeating the laziness this constructor is meant to preserve.
    pub unsafe fn from_controller(controller: Box<dyn AllocationController>, len: usize) -> Self {
        Self { controller, len }
    }

    /// Create a sequence of [Bytes] from the memory representation of an unknown type of elements.
    /// Prefer this over [`Self::from_elems`] when the datatype is not statically known and erased at runtime.
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
    /// In case the element type is not statically known at runtime, prefer to use [`Self::from_bytes_vec`].
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
    ///
    /// This never materializes lazy storage (see [`AllocationController::capacity`]).
    pub fn capacity(&self) -> usize {
        self.controller.capacity()
    }

    /// Convert the bytes back into a vector. This requires that the type has the same alignment as the element
    /// type this [Bytes] was initialized with.
    /// This only returns with Ok(_) if the conversion can be done without a memcopy
    pub fn try_into_vec<E: bytemuck::CheckedBitPattern + bytemuck::NoUninit>(
        mut self,
    ) -> Result<Vec<E>, Self> {
        // See if the length is compatible.
        // Use immutable validation to avoid triggering copy-on-write for SharedBytesAllocationController.
        // Note: This still calls memory() via Deref, which may trigger file I/O for FileAllocationController.
        let Ok(data) = bytemuck::checked::try_cast_slice::<_, E>(&self) else {
            return Err(self);
        };
        let length = data.len();
        // If so, try to convert the allocation to a vec. The data is already host-resident
        // here (the cast above went through `Deref`), so `capacity()` is free.
        let byte_capacity = self.controller.capacity();

        let Some(capacity) = byte_capacity.checked_div(size_of::<E>()) else {
            return Err(self);
        };
        if capacity * size_of::<E>() != byte_capacity {
            return Err(self);
        };
        // Vec::from_raw_parts requires that the pointer was allocated with
        // Layout::array::<E>(capacity). On drop, Vec deallocates with that
        // layout. If our allocation used a different alignment, the dealloc
        // layout won't match and that's UB per the GlobalAlloc contract:
        // https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html#safety-1
        if self.controller.alloc_align() != align_of::<E>() {
            return Err(self);
        }

        let Some(ptr) = self.controller.try_detach() else {
            return Err(self);
        };

        // SAFETY:
        // - ptr was allocated by the global allocator as per type-invariant
        // - alloc_align == align_of::<E> (checked above), so Vec will dealloc
        //   with the same layout as the original allocation.
        // - capacity * size_of::<E> == layout.size()
        // - 0 <= capacity
        // - length was computed from the bytemuck-ed slice into this allocation
        // - the layout represents a valid allocation, hence has allocation size less than isize::MAX
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
            let uninit_spare = &mut self.memory_mut_default()[len..new_cap];
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
    fn try_from_data(align: usize, data: &[u8]) -> Result<Self, AllocationError> {
        let controller = NativeAllocationController::alloc_with_data(Global, data, align)?;

        Ok(Self {
            controller: Box::new(controller),
            len: data.len(),
        })
    }

    /// Ensure the allocation's reported alignment is at least `align`, reallocating
    /// into a fresh controller if not. We check the controller's reported alignment
    /// (not the raw pointer) because downstream callers such as `try_into_vec::<E>`
    /// depend on `alloc_align()` matching the element alignment.
    fn try_enforce_runtime_align(&mut self, align: usize) -> Result<(), AllocationError> {
        if self.controller.alloc_align() >= align {
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
                // Keep the data in the same memory, for example the same arena.
                let new_controller = alloc_like(&*self.controller, new_cap, align)
                    .unwrap_or_else(|err| panic!("failed to grow bytes to {new_cap} bytes: {err}"));
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
        let memory = &self.memory_default()[0..self.len];
        // SAFETY: By construction, bytes up to len are initialized.
        unsafe { core::slice::from_raw_parts(memory.as_ptr().cast(), memory.len()) }
    }
}

impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = self.len;
        // SAFETY: We only expose this as initialized memory so cannot write uninitialized memory to this slice.
        let slice = unsafe { self.memory_mut_default() };
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
        if let Some(controller) = self.controller.duplicate() {
            return Self {
                controller,
                len: self.len,
            };
        }

        // The copy stays in the same memory as `self`, for example the same arena.
        self.copy_like(self)
            .unwrap_or_else(|err| panic!("failed to clone {} bytes: {err}", self.len))
    }
}

impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for Bytes {}

// The type is no-std; its tests need std (test_log, std collections).
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::{Bytes, SplitPolicy, ViewError};
    use alloc::{vec, vec::Vec};

    const _CONST_ASSERTS: fn() = || {
        fn test_send<T: Send>() {}
        fn test_sync<T: Sync>() {}
        test_send::<Bytes>();
        test_sync::<Bytes>();
    };

    fn test_serialization_roundtrip(bytes: &Bytes) {
        let mut serialized = Vec::new();
        ciborium::ser::into_writer(bytes, &mut serialized).expect("serialization to succeed");
        let roundtripped: Bytes = ciborium::de::from_reader(&mut serialized.as_slice())
            .expect("deserialization to succeed");
        assert_eq!(
            bytes, &roundtripped,
            "roundtripping through serialization didn't lead to equal Bytes"
        );
    }

    #[test_log::test]
    fn test_serialization() {
        test_serialization_roundtrip(&Bytes::from_elems::<i32>(vec![]));
        test_serialization_roundtrip(&Bytes::from_elems(vec![0xdead, 0xbeaf]));
    }

    #[test_log::test]
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

    #[test_log::test]
    fn test_grow() {
        let mut bytes = Bytes::from_elems::<u8>(vec![]);
        bytes.extend_from_byte_slice(&[0, 1, 2, 3]);
        assert_eq!(bytes[..], [0, 1, 2, 3][..]);

        let mut bytes = Bytes::from_elems(vec![42u8; 4]);
        bytes.extend_from_byte_slice(&[0, 1, 2, 3]);
        assert_eq!(bytes[..], [42, 42, 42, 42, 0, 1, 2, 3][..]);
    }

    #[test_log::test]
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

    #[test_log::test]
    fn test_split_and_use() {
        let bytes = Bytes::from_elems(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        let (left, right) = bytes.split(4, SplitPolicy::Shared).unwrap();
        assert_eq!(&left[..], &[0, 1, 2, 3]);
        assert_eq!(&right[..], &[4, 5, 6, 7]);
        let left2 = left.clone();
        assert_eq!(&left2[..], &[0, 1, 2, 3]);
    }

    #[test_log::test]
    fn test_split_at_zero() {
        let bytes = Bytes::from_elems(vec![10u8, 20, 30, 40]);
        let (left, right) = bytes.split(0, SplitPolicy::Shared).unwrap();
        assert_eq!(left.len(), 0);
        assert_eq!(&right[..], &[10, 20, 30, 40]);
    }

    /// A native allocation with element alignment > 1 can't be detached into a
    /// `Vec<u8>`, so it used to fail to split. With [`SplitPolicy::Shared`] it
    /// now shares behind an `Arc`.
    #[test_log::test]
    fn test_split_native_over_aligned_shared() {
        let bytes = Bytes::from_elems(vec![0u32, 1, 2, 3]);
        let (left, right) = bytes.split(8, SplitPolicy::Shared).unwrap();
        assert_eq!(&left[..], &[0, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&right[..], &[2, 0, 0, 0, 3, 0, 0, 0]);
        // The shared halves stay usable after cloning.
        assert_eq!(&left.clone()[..], &left[..]);
    }

    /// [`SplitPolicy::Owned`] copies eagerly so each half is independently owned
    /// and can be mutated in place (no copy-on-write occurs on first write).
    #[test_log::test]
    fn test_split_native_over_aligned_owned() {
        let bytes = Bytes::from_elems(vec![0u32, 1, 2, 3]);
        let (mut left, mut right) = bytes.split(8, SplitPolicy::Owned).unwrap();
        left[0] = 9;
        right[0] = 8;
        assert_eq!(&left[..], &[9, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&right[..], &[8, 0, 0, 0, 3, 0, 0, 0]);
    }

    #[test_log::test]
    fn test_split_at_end() {
        let bytes = Bytes::from_elems(vec![10u8, 20, 30, 40]);
        let (left, right) = bytes.split(4, SplitPolicy::Owned).unwrap();
        assert_eq!(&left[..], &[10, 20, 30, 40]);
        assert_eq!(right.len(), 0);
    }

    /// A plain heap allocation has no zero-copy window, so `view` reports
    /// [`ViewError::Unsupported`] rather than silently copying. `view` only
    /// borrows, so the original stays usable.
    #[test_log::test]
    fn test_view_heap_unsupported() {
        let bytes = Bytes::from_elems(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        assert!(matches!(bytes.view(2, 5), Err(ViewError::Unsupported)));
        // The original is still readable.
        assert_eq!(&bytes[..], &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// Sharing first (the caller owns the data) makes a zero-copy `view`
    /// available; mutating the window then copies on write, leaving the original
    /// untouched.
    #[test_log::test]
    fn test_view_shared_then_mutate() {
        let shared = Bytes::from_elems(vec![0u32, 1, 2, 3]).shared();
        let mut view = shared.view(4, 12).unwrap();
        view[0] = 9;
        assert_eq!(&view[..], &[9, 0, 0, 0, 2, 0, 0, 0]);
        // The original is untouched by the mutation.
        assert_eq!(
            &shared[..],
            &[0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
        );
    }

    #[test_log::test]
    fn test_view_invalid_range() {
        let bytes = Bytes::from_elems(vec![10u8, 20, 30, 40]);
        // end past the length, and start > end, are both rejected.
        assert!(bytes.view(0, 5).is_err());
        assert!(bytes.view(3, 1).is_err());
    }

    /// `from_bytes_vec` enforces `MAX_ALIGN`, so converting the result to a Vec of
    /// any type whose alignment is `<= MAX_ALIGN` must succeed. We iterate so the
    /// test hits a range of underlying allocator addresses.
    #[test_log::test]
    fn test_from_bytes_vec_try_into_vec_aligned_type() {
        for _ in 0..64 {
            let bytes = Bytes::from_bytes_vec(vec![0u8; 16]);
            let vec: Vec<u128> = bytes
                .try_into_vec::<u128>()
                .expect("MAX_ALIGN-aligned bytes must convert to Vec<u128>");
            assert_eq!(vec.len(), 1);
        }
    }

    #[test_log::test]
    fn test_many_extends_with_growth() {
        let mut bytes = Bytes::from_elems::<u8>(vec![]);
        for i in 0u8..=255 {
            bytes.extend_from_byte_slice(&[i]);
        }
        assert_eq!(bytes.len(), 256);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[255], 255);
    }
}
