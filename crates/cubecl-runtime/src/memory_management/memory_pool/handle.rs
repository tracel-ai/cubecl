use crate::memory_management::MemoryHandle;
use alloc::sync::Arc;
use core::cell::Cell;

/// Managed Memory handle
#[derive(Debug)]
pub struct ManagedMemoryHandle {
    descriptor: Arc<ManagedMemoryDescriptor>,
    // Holds only the reference counts of the handle.
    handle_count: Arc<()>,
}

/// Binding of a memory handle
#[derive(Debug)]
pub struct ManagedMemoryBinding {
    descriptor: Arc<ManagedMemoryDescriptor>,
}

impl Clone for ManagedMemoryHandle {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
            handle_count: self.handle_count.clone(),
        }
    }
}

/// Managed memory descriptor.
///
/// The location is wrapped in `Cell` for interior mutability: multiple
/// handles share the same descriptor via `Arc`, yet the memory management
/// system needs to update the location after creation (e.g. during
/// `reserve` / `bind`). All mutation happens on a single device thread,
/// so `Cell` is safe — we just need `unsafe impl Sync` because `Cell`
/// is `!Sync`.
///
/// An alternative would be `spin::Mutex<MemoryLocation>` which avoids the
/// `unsafe impl Sync` at the cost of a lock on every access.
pub(crate) struct ManagedMemoryDescriptor {
    pub(crate) id: ManagedMemoryId,
    location: Cell<MemoryLocation>,
}

// SAFETY: The channel requires ManagedMemoryHandle to be Send + Sync.
// Cell is _not_ Sync, but, we know that we only access this from the device thread,
// so we lie to the compiler and claim it is Sync. Other code must NOT rely on
// ManagedMemoryDescriptor being Send + Sync.
unsafe impl Send for ManagedMemoryDescriptor {}
unsafe impl Sync for ManagedMemoryDescriptor {}

impl core::fmt::Debug for ManagedMemoryDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ManagedMemoryDescriptor")
            .field("id", &self.id)
            .field("location", &self.location())
            .finish()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
/// Managed memory unique identifier.
pub struct ManagedMemoryId {
    pub(crate) value: usize,
}

impl PartialEq for ManagedMemoryDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ManagedMemoryDescriptor {}

#[derive(Clone, Copy, Debug)]
/// Defines where the [`ManagedMemoryId`] is located.
pub(crate) struct MemoryLocation {
    /// The memory pool index in the global memory management.
    pub pool: u8,
    /// The memory page index in a memory pool.
    pub page: u16,
    /// The memory slice index in a memory page.
    pub slice: u32,
    /// Whether the memory location is known/initialized.
    pub init: u8,
}

impl ManagedMemoryDescriptor {
    /// Update the memory location for the given [`ManagedMemoryId`].
    pub(crate) fn update_location(&self, location: MemoryLocation) {
        self.location.set(location);
    }

    /// Update only the slice position for the given [`ManagedMemoryId`].
    pub(crate) fn update_slice(&self, slice: u32) {
        self.location.update(|mut loc| {
            loc.slice = slice;
            loc
        });
    }

    /// Update only the memory page position for the given [`ManagedMemoryId`].
    pub fn update_page(&self, page: u16) {
        self.location.update(|mut loc| {
            loc.page = page;
            loc
        });
    }

    /// Retrieves the current location.
    pub(crate) fn location(&self) -> MemoryLocation {
        self.location.get()
    }

    pub(crate) fn slice(&self) -> usize {
        self.location.get().slice as usize
    }

    pub(crate) fn page(&self) -> usize {
        self.location.get().page as usize
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
            descriptor: Arc::new(ManagedMemoryDescriptor {
                id: ManagedMemoryId { value },
                location: Cell::new(MemoryLocation::uninit()),
            }),
            handle_count: Arc::new(()),
        }
    }

    /// Retrieves the descriptor for the current handle.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }

    /// Return whether the current handle can be modified in-place.
    pub fn can_mut(&self) -> bool {
        Arc::strong_count(&self.handle_count) <= 2
    }

    /// Return whether the current handle is free.
    pub fn is_free(&self) -> bool {
        Arc::strong_count(&self.descriptor) <= 1
    }

    /// Returns the binding for the current handle.
    pub fn binding(self) -> ManagedMemoryBinding {
        ManagedMemoryBinding {
            descriptor: self.descriptor.clone(),
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

impl ManagedMemoryBinding {
    /// Retrieves the descriptor for the current binding.
    pub(crate) fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }
}

impl Default for ManagedMemoryHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ManagedMemoryBinding {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
        }
    }
}

impl MemoryHandle<ManagedMemoryBinding> for ManagedMemoryHandle {
    fn can_mut(&self) -> bool {
        self.can_mut()
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
        handle1.descriptor().update_slice(4);
        assert_eq!(handle1.descriptor().slice(), 4);

        let handle2 = ManagedMemoryHandle::new();
        handle2
            .clone()
            .descriptor()
            .update_location(handle1.descriptor().location());
        assert_eq!(handle2.descriptor().slice(), 4);
    }

    #[test]
    fn test_location_visible_through_shared_arc() {
        let handle = ManagedMemoryHandle::new();
        let handle2 = handle.clone();

        let location = MemoryLocation::new(1, 2, 3);
        handle.descriptor().update_location(location);

        assert_eq!(handle2.descriptor().location().pool, 1);
        assert_eq!(handle2.descriptor().location().page, 2);
        assert_eq!(handle2.descriptor().location().slice, 3);
        assert_eq!(handle2.descriptor().location().init, 1);

        handle.descriptor().update_slice(42);
        assert_eq!(handle2.descriptor().slice(), 42);
    }
}
