use crate::id::{BindingRef, HandleRef};
use crate::memory_management::MemoryHandle;

/// Managed Memory handle
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedMemoryHandle {
    value: HandleRef<ManagedMemoryId>,
}

/// Managed memory id
#[derive(Debug)]
pub struct ManagedMemoryId {
    pub(crate) value: usize,
    pub(crate) location: MemoryLocation,
}

impl core::hash::Hash for ManagedMemoryId {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialEq for ManagedMemoryId {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl Eq for ManagedMemoryId {}

impl Clone for ManagedMemoryId {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            location: self.location.clone(),
        }
    }
}

#[derive(Clone, Debug)]
/// Defines where the [`ManagedMemoryId`] is located.
///
/// # Safety
///
/// The memory location should only be updated by an instance of [`super::super::MemoryManagement`].
///
/// Worse case:
///   - If there is an invalid write, it won't cause memory issue, only runtime errors.
pub(crate) struct MemoryLocation {
    /// The memory pool index in the global memory management.
    pub pool: u8,
    /// The memory page index in a memory pool.
    pub page: u16,
    /// The memory slice index in a memory page.
    pub slice: u32,
    /// Whether the memory location is known/intialized.
    pub init: u8,
}

impl ManagedMemoryId {
    /// Update the memory location for the given [`ManagedMemoryId`].
    pub(crate) fn update_location(&self, location: MemoryLocation) {
        let ptr = core::ptr::from_ref(&self.location) as *mut MemoryLocation;

        unsafe {
            ptr.write(location);
        }
    }

    /// Update only the slice position for the given [`ManagedMemoryId`].
    pub(crate) fn update_slice(&self, slice: u32) {
        let mut location = self.location.clone();
        location.slice = slice;
        self.update_location(location);
    }

    /// Update only the memory page position for the given [`ManagedMemoryId`].
    pub fn update_page(&self, page: u16) {
        let mut location = self.location.clone();
        location.page = page;
        self.update_location(location);
    }

    /// Retrieves the current location.
    pub(crate) fn location(&self) -> &MemoryLocation {
        &self.location
    }

    pub(crate) fn slice(&self) -> usize {
        self.location.slice as usize
    }

    pub(crate) fn page(&self) -> usize {
        self.location.page as usize
    }
}

impl MemoryLocation {
    /// Creates a new memory location.
    pub(crate) fn new(pool: u8, page: u16, slice: u32) -> Self {
        Self {
            pool,
            page,
            slice,
            init: 1,
        }
    }

    /// Creates a new uninitialized memory location.
    pub(crate) fn uninit() -> Self {
        Self {
            pool: 0,
            page: 0,
            slice: 0,
            init: 0,
        }
    }
}

impl ManagedMemoryHandle {
    /// Creates a new managed memory handle.
    pub fn new() -> Self {
        let value = Self::gen_id();
        Self {
            value: crate::id::HandleRef::new(ManagedMemoryId {
                value,
                location: MemoryLocation::uninit(),
            }),
        }
    }

    fn gen_id() -> usize {
        static COUNTER: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        let value = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if value == usize::MAX {
            core::panic!("Memory ID overflowed");
        }
        value
    }
}
impl core::ops::Deref for ManagedMemoryHandle {
    type Target = crate::id::HandleRef<ManagedMemoryId>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl Default for ManagedMemoryHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[doc = r" Binding of a memory handle."]
#[derive(Clone, Debug)]
pub struct ManagedMemoryBinding {
    value: BindingRef<ManagedMemoryId>,
}
impl ManagedMemoryHandle {
    /// Returns the binding for the current handle.
    pub fn binding(self) -> ManagedMemoryBinding {
        ManagedMemoryBinding {
            value: self.value.clone().binding(),
        }
    }
}
impl core::ops::Deref for ManagedMemoryBinding {
    type Target = BindingRef<ManagedMemoryId>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl MemoryHandle<ManagedMemoryBinding> for ManagedMemoryHandle {
    fn can_mut(&self) -> bool {
        HandleRef::can_mut(self)
    }

    fn binding(self) -> ManagedMemoryBinding {
        self.binding()
    }
}

/// Calculates a best-effort heuristic for the alignment of row-aligned tensors.
/// Prefers contiguous alignments for unit dimensions, 16-byte minimum alignment for non-unit,
/// scaling with input size up to `buffer_align`.
pub fn optimal_align(shape: usize, elem_size: usize, buffer_align: usize) -> usize {
    if shape == 1 {
        elem_size
    } else {
        (shape * elem_size)
            .next_power_of_two()
            .clamp(16, buffer_align)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_id_mutability() {
        let handle1 = ManagedMemoryHandle::new();
        handle1.id().update_slice(4);
        assert_eq!(handle1.id().slice(), 4);

        let handle2 = ManagedMemoryHandle::new();
        handle2
            .clone()
            .id()
            .update_location(handle1.id().location().clone());
        assert_eq!(handle2.id().slice(), 4);
    }
}
