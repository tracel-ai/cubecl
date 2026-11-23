use core::sync::atomic::{AtomicU64, Ordering};

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
    pub(super) location: AtomicU64,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

    /// Whether a reservation ever gave this memory a location.
    ///
    /// A handle is minted before its memory exists, and the reservation that
    /// follows can fail: a full device, a size no pool accepts.
    #[doc(hidden)]
    pub fn is_allocated(&self) -> bool {
        self.location().init != 0
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
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                Some(update(MemoryLocation::from_bits(bits)).to_bits())
            });
    }
}

impl MemoryLocation {
    /// The location packed into one word, so it can live in an atomic: pool
    /// in the low byte, then page, then slice, then the init flag on top.
    pub(super) fn to_bits(self) -> u64 {
        self.pool as u64
            | (self.page as u64) << 8
            | (self.slice as u64) << 24
            | (self.init as u64) << 56
    }

    pub(super) fn from_bits(bits: u64) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;

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
