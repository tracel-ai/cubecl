use crate::memory_management::MemoryHandle;
use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};

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

/// A list of bindings that are shared across multiple streams.
#[derive(Debug, Default)]
pub struct SharedMemoryBindings {
    /// The bindings.
    pub bindings: Vec<ManagedMemoryBinding>,
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
/// Multiple handles share the same descriptor via `Arc`, yet the memory
/// management system needs to update the location after creation (e.g. during
/// `reserve` / `bind`). The location is packed into an atomic for that.
///
/// All mutation happens on the device thread, so `Relaxed` is all the ordering
/// it needs — and on the targets that matter that is a plain load and store.
/// Being atomic rather than a `Cell` is what makes the descriptor `Sync` by
/// construction: these methods are public so the pools in `cubecl-server` can
/// reach them, and anything public can be called from any thread.
#[doc(hidden)]
pub struct ManagedMemoryDescriptor {
    #[doc(hidden)]
    pub id: ManagedMemoryId,
    location: AtomicU64,
}

impl core::fmt::Debug for ManagedMemoryDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ManagedMemoryDescriptor")
            .field("id", &self.id)
            .field("location", &self.location())
            .finish()
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
/// Managed memory unique identifier.
pub struct ManagedMemoryId {
    #[doc(hidden)]
    pub value: usize,
}

impl PartialEq for ManagedMemoryDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ManagedMemoryDescriptor {}

#[derive(Clone, Copy, Debug)]
/// Defines where the [`ManagedMemoryId`] is located.
#[doc(hidden)]
pub struct MemoryLocation {
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
    #[doc(hidden)]
    pub fn update_location(&self, location: MemoryLocation) {
        self.location.store(location.to_bits(), Ordering::Relaxed);
    }

    /// Update only the slice position for the given [`ManagedMemoryId`].
    #[doc(hidden)]
    pub fn update_slice(&self, slice: u32) {
        self.modify(|location| MemoryLocation { slice, ..location });
    }

    /// Update only the memory page position for the given [`ManagedMemoryId`].
    #[doc(hidden)]
    pub fn update_page(&self, page: u16) {
        self.modify(|location| MemoryLocation { page, ..location });
    }

    /// Retrieves the current location.
    #[doc(hidden)]
    pub fn location(&self) -> MemoryLocation {
        MemoryLocation::from_bits(self.location.load(Ordering::Relaxed))
    }

    #[doc(hidden)]
    pub fn slice(&self) -> usize {
        self.location().slice as usize
    }

    #[doc(hidden)]
    pub fn page(&self) -> usize {
        self.location().page as usize
    }

    fn modify(&self, update: impl Fn(MemoryLocation) -> MemoryLocation) {
        // Never `Err`: the closure always has an update to make.
        let _ = self
            .location
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                Some(update(MemoryLocation::from_bits(bits)).to_bits())
            });
    }
}

impl MemoryLocation {
    /// The location packed into one word, so it can live in an atomic: pool
    /// in the low byte, then page, then slice, then the init flag on top.
    fn to_bits(self) -> u64 {
        self.pool as u64
            | (self.page as u64) << 8
            | (self.slice as u64) << 24
            | (self.init as u64) << 56
    }

    fn from_bits(bits: u64) -> Self {
        Self {
            pool: bits as u8,
            page: (bits >> 8) as u16,
            slice: (bits >> 24) as u32,
            init: (bits >> 56) as u8,
        }
    }

    /// Creates a new memory location.
    #[doc(hidden)]
    pub fn new(pool: u8, page: u16, slice: u32) -> Self {
        Self {
            pool,
            page,
            slice,
            init: 1,
        }
    }

    /// Creates a new uninitialized memory location.
    #[doc(hidden)]
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
            descriptor: Arc::new(ManagedMemoryDescriptor {
                id: ManagedMemoryId { value },
                location: AtomicU64::new(MemoryLocation::uninit().to_bits()),
            }),
            handle_count: Arc::new(()),
        }
    }

    /// Retrieves the descriptor for the current handle.
    #[doc(hidden)]
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
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
    #[doc(hidden)]
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }

    /// The id of the memory this binding is bound to, stable for as long as the
    /// allocation lives and never reused by a later one.
    pub fn id(&self) -> ManagedMemoryId {
        self.descriptor.id
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

impl SharedMemoryBindings {
    /// Clears the shared bindings list.
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Returns true if the shared bindings list is empty.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Push a memory binding to the list of shared bindings.
    pub fn push(&mut self, binding: ManagedMemoryBinding) {
        self.bindings.push(binding)
    }
}

impl cubecl_common::pool::Reclaim for SharedMemoryBindings {
    fn reclaim(&mut self) {
        self.clear();
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

    /// Every field gets its own bits in the packed word, so none bleeds into
    /// its neighbour even at its widest.
    #[test]
    fn a_location_survives_packing_at_every_extreme() {
        let fields = |location: MemoryLocation| {
            (location.pool, location.page, location.slice, location.init)
        };

        for location in [
            MemoryLocation::uninit(),
            MemoryLocation::new(1, 2, 3),
            MemoryLocation::new(u8::MAX, u16::MAX, u32::MAX),
            MemoryLocation {
                init: u8::MAX,
                ..MemoryLocation::uninit()
            },
        ] {
            let packed = MemoryLocation::from_bits(location.to_bits());

            assert_eq!(fields(packed), fields(location));
        }
    }
}
