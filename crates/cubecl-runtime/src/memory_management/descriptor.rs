use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Which memory a [`ManagedMemoryHandle`](super::ManagedMemoryHandle) names,
/// and where in the pools that memory currently lives.
///
/// Every handle and binding of one allocation shares one descriptor, and the
/// pools move the allocation after the handle exists (on `reserve`, `bind` or a
/// relocation), so the location is updated through a shared reference.
///
/// All mutation happens on the device thread, so `Relaxed` is all the ordering
/// it needs, and on the targets that matter that is a plain load and store. The
/// one exception is a lazy read let go from the thread that drops it: the count
/// only rises on the device thread, so a value it reads late costs that thread
/// a lookup and never misses a read.
/// Being atomic rather than a `Cell` is what makes the descriptor `Sync` by
/// construction: these methods are public so the pools in `cubecl-server` can
/// reach them, and anything public can be called from any thread.
#[doc(hidden)]
pub struct ManagedMemoryDescriptor {
    id: ManagedMemoryId,
    location: AtomicU64,
    /// The lazy reads of the memory a write must start before it runs: how many its device
    /// keeps, which spares every write to memory without one the device's own lookup.
    lazy_reads: AtomicU32,
    /// Whether kernels on other devices may bind the memory, which write it with nothing this
    /// device can run first.
    shared_with_peers: AtomicBool,
}

/// The id of one allocation, never given to another.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct ManagedMemoryId {
    #[doc(hidden)]
    pub value: usize,
}

/// Where an allocation lives: its pool, its page in that pool and its slice in
/// that page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
    /// A descriptor with a fresh id and no location yet.
    #[doc(hidden)]
    pub fn new() -> Self {
        Self {
            id: next_id(),
            location: AtomicU64::new(MemoryLocation::uninit().to_bits()),
            lazy_reads: AtomicU32::new(0),
            shared_with_peers: AtomicBool::new(false),
        }
    }

    /// The id of the allocation, stable for as long as it lives.
    #[doc(hidden)]
    pub fn id(&self) -> ManagedMemoryId {
        self.id
    }

    /// Moves the allocation to `location`, seen at once by every handle and
    /// binding of it.
    #[doc(hidden)]
    pub fn update_location(&self, location: MemoryLocation) {
        self.location.store(location.to_bits(), Ordering::Relaxed);
    }

    /// Moves the allocation to another slice of the same page.
    #[doc(hidden)]
    pub fn update_slice(&self, slice: u32) {
        self.modify(|location| MemoryLocation { slice, ..location });
    }

    /// Moves the allocation to another page of the same pool, at the same
    /// slice index.
    #[doc(hidden)]
    pub fn update_page(&self, page: u16) {
        self.modify(|location| MemoryLocation { page, ..location });
    }

    /// Where the allocation lives now.
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

    /// The index of the allocation's slice in its page.
    #[doc(hidden)]
    pub fn slice(&self) -> usize {
        self.location().slice as usize
    }

    /// The index of the allocation's page in its pool.
    #[doc(hidden)]
    pub fn page(&self) -> usize {
        self.location().page as usize
    }

    /// Whether its device keeps a lazy read of the memory that a write must start first.
    #[doc(hidden)]
    pub fn has_lazy_reads(&self) -> bool {
        self.lazy_reads.load(Ordering::Relaxed) != 0
    }

    /// Its device keeps one more lazy read of the memory.
    #[doc(hidden)]
    pub fn lazy_read_kept(&self) {
        self.lazy_reads.fetch_add(1, Ordering::Relaxed);
    }

    /// Its device keeps one lazy read of the memory fewer.
    #[doc(hidden)]
    pub fn lazy_read_let_go(&self) {
        self.lazy_reads.fetch_sub(1, Ordering::Relaxed);
    }

    /// Kernels on other devices may bind the memory from now on.
    #[doc(hidden)]
    pub fn share_with_peers(&self) {
        self.shared_with_peers.store(true, Ordering::Relaxed);
    }

    /// Whether kernels on other devices may bind the memory.
    #[doc(hidden)]
    pub fn is_shared_with_peers(&self) -> bool {
        self.shared_with_peers.load(Ordering::Relaxed)
    }
}

impl ManagedMemoryDescriptor {
    fn modify(&self, update: impl Fn(MemoryLocation) -> MemoryLocation) {
        // Never `Err`: `map_bits` always has an update to make.
        let _ = self
            .location
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                MemoryLocation::map_bits(bits, &update)
            });
    }
}

impl Default for ManagedMemoryDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for ManagedMemoryDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        (self.id, self.location()).fmt(f)
    }
}

impl PartialEq for ManagedMemoryDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ManagedMemoryDescriptor {}

impl MemoryLocation {
    /// A location in `pool`, `page` and `slice`, marked as known.
    #[doc(hidden)]
    pub fn new(pool: u8, page: u16, slice: u32) -> Self {
        Self {
            pool,
            page,
            slice,
            init: 1,
        }
    }

    /// The location of memory no reservation has placed yet.
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

    /// The packed location `update` makes of the packed location `bits`, in
    /// the shape an atomic `try_update` takes.
    fn map_bits(bits: u64, update: impl Fn(Self) -> Self) -> Option<u64> {
        Some(update(Self::from_bits(bits)).to_bits())
    }
}

/// A [`ManagedMemoryId`] no earlier call returned.
fn next_id() -> ManagedMemoryId {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let value = COUNTER.fetch_add(1, Ordering::Relaxed);
    assert!(value != usize::MAX, "Memory ids should not run out");
    ManagedMemoryId { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_descriptor_is_not_allocated() {
        let descriptor = ManagedMemoryDescriptor::new();

        assert!(!descriptor.is_allocated());
    }

    #[test]
    fn a_placed_descriptor_is_allocated() {
        let descriptor = ManagedMemoryDescriptor::new();

        descriptor.update_location(MemoryLocation::new(0, 0, 0));

        assert!(descriptor.is_allocated());
    }

    #[test]
    fn two_descriptors_never_share_an_id() {
        let (first, second) = (
            ManagedMemoryDescriptor::new(),
            ManagedMemoryDescriptor::new(),
        );

        assert_ne!(first.id(), second.id());
    }

    #[test]
    fn an_updated_location_is_read_back() {
        let descriptor = ManagedMemoryDescriptor::new();

        descriptor.update_location(MemoryLocation::new(1, 2, 3));

        assert_eq!(descriptor.location(), MemoryLocation::new(1, 2, 3));
    }

    #[test]
    fn updating_the_slice_keeps_pool_and_page() {
        let descriptor = ManagedMemoryDescriptor::new();
        descriptor.update_location(MemoryLocation::new(1, 2, 3));

        descriptor.update_slice(42);

        assert_eq!(descriptor.location(), MemoryLocation::new(1, 2, 42));
    }

    #[test]
    fn updating_the_page_keeps_pool_and_slice() {
        let descriptor = ManagedMemoryDescriptor::new();
        descriptor.update_location(MemoryLocation::new(1, 2, 3));

        descriptor.update_page(7);

        assert_eq!(descriptor.location(), MemoryLocation::new(1, 7, 3));
    }

    #[test]
    fn slice_and_page_read_the_location() {
        let descriptor = ManagedMemoryDescriptor::new();
        descriptor.update_location(MemoryLocation::new(1, 2, 3));

        assert_eq!((descriptor.page(), descriptor.slice()), (2, 3));
    }

    /// Every field gets its own bits in the packed word, so none bleeds into
    /// its neighbour even at its widest.
    #[test]
    fn a_location_survives_packing_at_every_extreme() {
        for location in [
            MemoryLocation::uninit(),
            MemoryLocation::new(1, 2, 3),
            MemoryLocation::new(u8::MAX, u16::MAX, u32::MAX),
            MemoryLocation {
                init: u8::MAX,
                ..MemoryLocation::uninit()
            },
        ] {
            assert_eq!(MemoryLocation::from_bits(location.to_bits()), location);
        }
    }
}
