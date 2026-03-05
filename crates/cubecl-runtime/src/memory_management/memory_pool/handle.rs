use crate::id::HandleRef;
use crate::memory_management::MemoryHandle;
use crate::server::{Binding, MemorySlot};
use cubecl_common::stream_id::StreamId;

/// Managed Memory handle
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedMemoryHandle {
    value: HandleRef<ManagedMemoryId>,
}

/// Managed memory id
#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub(crate) struct MemoryLocation {
    pub pool: u8,
    pub page: u16,
    pub slice: u32,
    pub init: u8,
}

/// # Safety
///
/// The memory location should only be updated from memory management, and worse case if someones
/// write wrong values into the location, it won't cause memory issue, only runtime errors.
impl ManagedMemoryId {
    pub fn update_location(&self, location: MemoryLocation) {
        let ptr = core::ptr::from_ref(&self.location) as *mut MemoryLocation;

        unsafe {
            ptr.write(location);
        }
    }

    pub fn update_slice(&self, slice: u32) {
        let mut location = self.location.clone();
        location.slice = slice;
        self.update_location(location);
    }

    pub fn update_page(&self, page: u16) {
        let mut location = self.location.clone();
        location.page = page;
        self.update_location(location);
    }

    pub fn location(&self) -> &MemoryLocation {
        &self.location
    }

    pub fn location_mut(&mut self) -> &mut MemoryLocation {
        &mut self.location
    }

    pub fn slice(&self) -> usize {
        self.location.slice as usize
    }
    pub fn page(&self) -> usize {
        self.location.page as usize
    }
    pub fn pool(&self) -> usize {
        self.location.pool as usize
    }
}

impl MemoryLocation {
    /// Creates a new memory location.
    pub fn new(pool: u8, page: u16, slice: u32) -> Self {
        Self {
            pool,
            page,
            slice,
            init: 1,
        }
    }

    /// Creates a new uninitialized memory location.
    pub fn uninit() -> Self {
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
    value: crate::id::BindingRef<ManagedMemoryId>,
}
impl ManagedMemoryHandle {
    /// Returns the binding for the current handle.
    pub fn binding(self) -> ManagedMemoryBinding {
        ManagedMemoryBinding {
            value: self.value.binding(),
        }
    }
}
impl core::ops::Deref for ManagedMemoryBinding {
    type Target = crate::id::BindingRef<ManagedMemoryId>;
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

impl ManagedMemoryHandle {
    /// Converts the current managed memory handle to a memory slot given a handle.
    pub fn into_slot(self, binding: &Binding, cursor: u64, stream: StreamId) -> MemorySlot {
        // We ignore the offsets from the handle, since those are resolved later when we use
        // the memory slot.
        MemorySlot {
            memory: self,
            offset_start: None,
            offset_end: None,
            cursor,
            stream,
            size: binding.size(),
        }
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
