use super::{AccessError, AccessPolicy, AllocationController, AllocationError, AllocationProperty};
use alloc::alloc::{Layout, LayoutError, alloc_zeroed, dealloc};
use core::mem::MaybeUninit;
use core::ptr::NonNull;

/// Owns zeroed memory allocated at an alignment of its own, freed when it drops.
pub struct AlignedAllocationController {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedAllocationController {
    /// `None` when the host cannot allocate `layout`.
    pub fn zeroed(layout: Layout) -> Option<Self> {
        let ptr = match layout.size() {
            // Never allocated, so never freed: an empty allocation only needs an aligned address.
            0 => NonNull::new(layout.align() as *mut u8)?,
            // SAFETY: the layout's size is not zero.
            _ => NonNull::new(unsafe { alloc_zeroed(layout) })?,
        };
        Some(Self { ptr, layout })
    }

    /// A copy of `data`, its first byte aligned to `align`.
    pub fn with_data(data: &[u8], align: usize) -> Result<Self, LayoutError> {
        let layout = Layout::from_size_align(data.len(), align)?;
        let controller =
            Self::zeroed(layout).unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout));
        // SAFETY: the allocation holds `data.len()` bytes, and no other reference to it exists.
        unsafe { core::slice::from_raw_parts_mut(controller.ptr.as_ptr(), data.len()) }
            .copy_from_slice(data);
        Ok(controller)
    }
}

impl AllocationController for AlignedAllocationController {
    fn alloc_align(&self) -> usize {
        self.layout.align()
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Native
    }

    fn grow(&mut self, size: usize, align: usize) -> Result<(), AllocationError> {
        let align = align.max(self.layout.align());
        let layout =
            Layout::from_size_align(size, align).map_err(|_| AllocationError::OutOfMemory)?;
        let grown = Self::zeroed(layout).ok_or(AllocationError::OutOfMemory)?;
        // SAFETY: both allocations are live, distinct, and `grown` holds at least as many bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.ptr.as_ptr(),
                grown.ptr.as_ptr(),
                self.layout.size(),
            )
        };
        *self = grown;
        Ok(())
    }

    fn may_alias_mutable_storage(&self) -> bool {
        false
    }

    unsafe fn memory_mut(
        &mut self,
        _policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        // SAFETY: `ptr` holds `layout.size()` bytes for as long as `self` lives, and `&mut self`
        // excludes every other reader.
        Ok(
            unsafe {
                core::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.layout.size())
            },
        )
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        // SAFETY: `ptr` holds `layout.size()` bytes for as long as `self` lives.
        Ok(unsafe { core::slice::from_raw_parts(self.ptr.as_ptr().cast(), self.layout.size()) })
    }
}

impl Drop for AlignedAllocationController {
    fn drop(&mut self) {
        if self.layout.size() == 0 {
            return;
        }
        // SAFETY: `ptr` was allocated with `layout` and is freed once, here.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}
