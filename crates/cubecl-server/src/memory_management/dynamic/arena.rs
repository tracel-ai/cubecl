//! The pools a workload's allocations are carved from, one per page size it
//! grew through.

use crate::{
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryPoolKind, MemoryPoolReport, MemoryUsage,
        memory_pool::{MemoryPool, PageMapping, SlicedPool, calculate_padding},
        relocation::{
            Landed, LiveAllocation, Move, OutdatedPages, Relocation, StorageCopy, TargetRoom,
        },
    },
    server::IoError,
    storage::ComputeStorage,
};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::MemoryDeviceProperties;

/// How many page sizes the arena holds at once.
///
/// A slot frees once the pool in it drains, and a relocation drains the
/// outdated ones, so a workload reaches this only by growing through that many
/// sizes while each keeps something alive.
pub const ARENA_SLOTS: usize = 64;

/// Slack a page keeps past the largest allocation it was sized for, so that
/// allocation still fits once aligned.
const PAGE_SLACK: u64 = 1024 * 1024;

/// The unit a page size is rounded up to.
const PAGE_GRANULE: u64 = 1024 * 1024;

/// A fixed number of sliced pools, one per page size a workload's allocations
/// grew through.
///
/// The *current* pool carves every new allocation. When one outgrows its
/// pages, a pool of the new size takes over and the old one is *outdated*: it
/// serves no new reservation, gives each page back as it empties, and frees
/// its slot once it holds none.
pub struct PoolArena {
    /// Indexed by slot; a slot is empty once its pool drained.
    slots: Vec<Option<SlicedPool>>,
    /// The pool index a slice's location carries for slot 0.
    first_index: u8,
    /// The slot new allocations are carved from.
    current: usize,
    /// The slots a growth left behind, draining.
    outdated: Vec<usize>,
    sizing: PageSizing,
}

/// What a [`PoolArena`] holds, as far as a relocation plan can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaShape {
    current_pages: u64,
    outdated_pools: usize,
    outdated_pages: u64,
    outdated_guarded: usize,
}

/// How the pages of a [`PoolArena`] are sized.
#[derive(Debug, Clone, Copy)]
pub struct PageSizing {
    /// The smallest page, so a workload of small allocations still carves
    /// pages worth carving.
    min_page_size: u64,
    /// The largest page the device allocates, alignment-rounded down: an
    /// allocation no page of that size fits is not the arena's to serve.
    max_page_size: u64,
    alignment: u64,
}

/// How an allocation fits the pages the arena carves now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// A page of the current size holds it.
    Fits,
    /// It needs pages of `page_size`, which outdates the current ones.
    Grow {
        /// The size the new pages are allocated at.
        page_size: u64,
    },
    /// No page the device allocates holds it.
    TooLarge,
}

impl PageSizing {
    /// Pages from `min_page_size` up to the largest the device with
    /// `properties` allocates.
    pub fn new(min_page_size: u64, properties: &MemoryDeviceProperties) -> Self {
        let alignment = properties.alignment;
        let max_page_size = (properties.max_page_size / alignment * alignment).max(alignment);
        Self {
            min_page_size: min_page_size
                .max(alignment)
                .next_multiple_of(alignment)
                .min(max_page_size),
            max_page_size,
            alignment,
        }
    }

    /// How an allocation of `size` bytes fits pages of `page_size`.
    ///
    /// A growth sizes the new pages to the allocation, a megabyte of slack
    /// on top, rounded to the megabyte: never below the floor, never past
    /// the device's largest page.
    pub fn fit(&self, size: u64, page_size: u64) -> Fit {
        let needed = size + calculate_padding(size, self.alignment);
        if needed <= page_size {
            return Fit::Fits;
        }
        if needed > self.max_page_size {
            return Fit::TooLarge;
        }
        Fit::Grow {
            page_size: size
                .saturating_add(PAGE_SLACK)
                .next_multiple_of(PAGE_GRANULE)
                .next_multiple_of(self.alignment)
                .clamp(self.min_page_size, self.max_page_size),
        }
    }

    /// A pool carving pages of `page_size`, whose slices carry `index`.
    fn pool(&self, page_size: u64, index: u8) -> SlicedPool {
        // It takes every size it is handed, so nothing is refused for being
        // larger than a slice of a page it fits.
        SlicedPool::new(page_size, page_size, self.alignment, index)
    }
}

impl PoolArena {
    /// An arena whose first pool carves the smallest pages `sizing` allows,
    /// its slots addressed from `first_index` on.
    pub fn new(sizing: PageSizing, first_index: u8) -> Self {
        let mut slots = Vec::with_capacity(ARENA_SLOTS);
        slots.push(Some(sizing.pool(sizing.min_page_size, first_index)));
        slots.resize_with(ARENA_SLOTS, || None);
        Self {
            slots,
            first_index,
            current: 0,
            outdated: Vec::with_capacity(ARENA_SLOTS),
            sizing,
        }
    }

    /// The pool `index` names, while one is there.
    pub fn pool(&self, index: usize) -> Option<&SlicedPool> {
        let slot = index.checked_sub(self.first_index as usize)?;
        self.slots.get(slot)?.as_ref()
    }

    /// The pool `index` names, mutably.
    pub fn pool_mut(&mut self, index: usize) -> Option<&mut SlicedPool> {
        let slot = index.checked_sub(self.first_index as usize)?;
        self.slots.get_mut(slot)?.as_mut()
    }

    /// The pool new allocations are carved from.
    pub fn current(&self) -> &SlicedPool {
        self.slots[self.current]
            .as_ref()
            .expect("the current slot always holds a pool")
    }

    /// The pool new allocations are carved from, mutably.
    pub fn current_mut(&mut self) -> &mut SlicedPool {
        self.slots[self.current]
            .as_mut()
            .expect("the current slot always holds a pool")
    }

    /// The pools a growth left behind.
    pub fn outdated(&self) -> impl Iterator<Item = &SlicedPool> {
        self.outdated
            .iter()
            .filter_map(|slot| self.slots[*slot].as_ref())
    }

    /// Whether any pool is outdated.
    pub fn has_outdated(&self) -> bool {
        !self.outdated.is_empty()
    }

    /// What a relocation plans from: how many pages the current pool has
    /// room on, and what the outdated pools hold and keep guarded. A plan that
    /// found nothing to move finds nothing again until this changes.
    ///
    /// Room freed inside the current pages does not show here, since slices
    /// free as their owners drop them: that room is found by the plan after
    /// the next page, growth or explicit cleanup.
    pub fn shape(&self) -> ArenaShape {
        ArenaShape {
            current_pages: self.current().pages_held(),
            outdated_pools: self.outdated.len(),
            outdated_pages: self.outdated().map(SlicedPool::pages_held).sum(),
            outdated_guarded: self
                .outdated()
                .flat_map(SlicedPool::pages)
                .filter(|page| page.is_guarded())
                .count(),
        }
    }

    /// Whether every slot holds a pool, so the next growth has nowhere to go.
    pub fn is_full(&self) -> bool {
        self.slots.iter().all(Option::is_some)
    }

    /// The pools `source` and `target` name, which are distinct.
    pub fn pair_mut(
        &mut self,
        source: usize,
        target: usize,
    ) -> Option<(&mut SlicedPool, &mut SlicedPool)> {
        let first = self.first_index as usize;
        let [source, target] = self
            .slots
            .get_disjoint_mut([source.checked_sub(first)?, target.checked_sub(first)?])
            .ok()?;
        Some((source.as_mut()?, target.as_mut()?))
    }

    /// Reserve `size` bytes on the current pool, growing the pages when `size`
    /// outgrows them.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no page the device allocates fits it,
    /// [`IoError::PageSizesExhausted`] when it needs a new page size and
    /// every slot is taken, and whatever the device refused.
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match self.sizing.fit(size, self.current().page_size()) {
            Fit::Fits => self.reserve_current(storage, size, mapping, failures),
            Fit::Grow { page_size } => self.grow(storage, size, page_size, mapping, failures),
            Fit::TooLarge => Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// Reserve `size` bytes in the room the current pool holds. `None` when
    /// it has none for it, or when it would need larger pages.
    pub fn try_reserve(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        match self.sizing.fit(size, self.current().page_size()) {
            Fit::Fits => self.current_mut().try_reserve(size, failures),
            Fit::Grow { .. } | Fit::TooLarge => None,
        }
    }

    /// Release what the current pool no longer needs, and drop every outdated
    /// pool that has drained.
    pub fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
        failures: &mut ErrorGraph,
    ) {
        self.current_mut()
            .cleanup(storage, alloc_nr, explicit, failures);
        self.drain(storage, failures);
    }

    /// Give every empty page of the outdated pools back, and free the slot of
    /// each pool left with none.
    ///
    /// An outdated pool returns a page the moment it empties, rather than
    /// waiting for the explicit cleanup the current pool waits for.
    pub fn drain<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        let slots = &mut self.slots;
        self.outdated.retain(|&slot| {
            let Some(pool) = slots[slot].as_mut() else {
                return false;
            };
            pool.release_empty(storage, failures);
            let drained = pool.is_empty();
            if drained {
                slots[slot] = None;
            }
            !drained
        });
    }

    /// The usage of every pool held.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.slots
            .iter()
            .flatten()
            .fold(MemoryUsage::default(), |usage, pool| {
                usage.combine(pool.get_memory_usage())
            })
    }

    /// A report for the current pool, stating how many pages the outdated
    /// ones hold, then one per outdated pool.
    pub fn report(&self) -> impl Iterator<Item = MemoryPoolReport> + '_ {
        let current = self.current();
        let kind = MemoryPoolKind::Adaptive {
            page_size: current.page_size(),
            outdated_pages: self.outdated().map(SlicedPool::pages_held).sum(),
        };
        core::iter::once(current.report(kind))
            .chain(self.outdated().map(|pool| pool.report(pool.kind())))
    }

    /// Reserve a slice on the current pool for every live allocation on the
    /// outdated pages [the analysis](OutdatedPages::plan) plans to empty, so
    /// those pools drain instead of waiting on their longest-lived slice.
    ///
    /// `room` says whether a target may take a new page when the room held
    /// runs out. Nothing moves yet: the caller copies the bytes and hands the
    /// [`Landed`] relocation back to [`commit`](Self::commit).
    pub fn plan<Storage: ComputeStorage>(
        &mut self,
        room: TargetRoom,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> Relocation {
        let plan = OutdatedPages::new(self.outdated().flat_map(SlicedPool::pages)).plan();

        let mut moves = Vec::new();
        for page in plan {
            // A page whose allocations do not all find a target keeps none of
            // them: moving part of what is on it frees nothing. A page later in
            // the plan may still fit what this one did not.
            let start = moves.len();
            for allocation in page.live {
                match self.plan_move(allocation, room, storage, failures) {
                    Some(relocated) => moves.push(relocated),
                    None => {
                        moves.truncate(start);
                        break;
                    }
                }
            }
        }
        Relocation::new(moves)
    }

    /// Hand every allocation of `landed` over to the target reserved for it,
    /// now that its bytes are there, and drop the outdated pools that emptied.
    pub fn commit<Storage: ComputeStorage>(
        &mut self,
        landed: Landed,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        for Move {
            allocation, target, ..
        } in landed.into_moves()
        {
            // Locations are read now, not at planning: a pool that released
            // pages since may have renumbered them.
            let source = allocation.descriptor().location();
            let destination = target.descriptor().location();
            // Only the slices hold the handles once these go.
            drop(target);
            drop(allocation);

            if let Some((source_pool, target_pool)) =
                self.pair_mut(source.pool as usize, destination.pool as usize)
            {
                source_pool
                    .slice_at(source)
                    .hand_over(target_pool.slice_at(destination), failures);
            }
        }
        self.drain(storage, failures);
    }

    /// Reserve a target for `allocation` on the current pool, and say what has
    /// to be copied into it. `None` when `room` has no place for it.
    fn plan_move<Storage: ComputeStorage>(
        &mut self,
        allocation: LiveAllocation,
        room: TargetRoom,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> Option<Move> {
        let source = allocation.handle.descriptor().location();
        let pool = self.pool_mut(source.pool as usize)?;
        let source_storage = pool.slice_at(source).storage.clone();
        let source_mapped = pool.is_mapped_at(source);

        let pool = self.current_mut();
        let target = match (pool.try_reserve(allocation.size, failures), room) {
            (Some(target), _) => target,
            (None, TargetRoom::Held) => return None,
            (None, TargetRoom::MayAllocate) => pool
                .alloc(storage, allocation.size, PageMapping::Eager, failures)
                .ok()?,
        };
        let destination = target.descriptor().location();
        // The bytes have to land somewhere real.
        if source_mapped {
            pool.map_page_at(storage, destination).ok()?;
        }
        let copy = source_mapped.then(|| StorageCopy {
            source: source_storage,
            target: pool.slice_at(destination).storage.clone(),
        });

        Some(Move {
            allocation: allocation.handle,
            target,
            copy,
        })
    }

    fn reserve_current<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let pool = self.current_mut();
        match pool.try_reserve(size, failures) {
            Some(handle) => Ok(handle),
            None => pool.alloc(storage, size, mapping, failures),
        }
    }

    /// Carve `size` bytes from a new pool of `page_size` pages, which takes
    /// over from the current one only once the device gave it its first page:
    /// a refused allocation leaves the page size where it was.
    fn grow<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        page_size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let Some(slot) = self.slots.iter().position(Option::is_none) else {
            return Err(IoError::PageSizesExhausted {
                size,
                backtrace: BackTrace::capture(),
            });
        };
        // The pages the current pool leaves empty go back with the next
        // cleanup, which drains the pool once it is outdated. Not here: a
        // reservation never releases a page, so a caller that must keep every
        // page where it is can still reserve.
        let mut pool = self.sizing.pool(page_size, self.first_index + slot as u8);
        let handle = pool.alloc(storage, size, mapping, failures)?;

        self.slots[slot] = Some(pool);
        self.outdated.push(self.current);
        self.current = slot;
        Ok(handle)
    }
}

impl core::fmt::Display for PoolArena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for pool in self.slots.iter().flatten() {
            write!(f, "{pool}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::relocation::CopyQueue;
    use crate::server::ServerError;
    use crate::storage::{StorageHandle, StorageId, StorageUtilization};

    const MIB: u64 = 1024 * 1024;

    fn sizing() -> PageSizing {
        PageSizing::new(2 * MIB, &MemoryDeviceProperties::new(16 * MIB, 32))
    }

    /// Device memory that is only bookkeeping, refusing any allocation past
    /// `limit` bytes the way a full device would.
    struct Bookkeeping {
        limit: u64,
    }

    impl ComputeStorage for Bookkeeping {
        type Resource = ();

        fn alignment(&self) -> usize {
            32
        }

        fn get(&mut self, _handle: &StorageHandle) -> Result<(), IoError> {
            Ok(())
        }

        fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
            if size > self.limit {
                return Err(IoError::OutOfMemory {
                    size,
                    backtrace: BackTrace::capture(),
                });
            }
            Ok(StorageHandle::new(
                StorageId::new(),
                StorageUtilization { offset: 0, size },
            ))
        }

        fn dealloc(&mut self, _id: StorageId) {}

        fn flush(&mut self) {}

        fn bytes_allocated(&self) -> u64 {
            0
        }
    }

    /// Copies that land the moment they are made.
    struct Landing;

    impl CopyQueue<Bookkeeping> for Landing {
        fn wait_device(&mut self) -> Result<(), ServerError> {
            Ok(())
        }

        fn copy(&mut self, _storage: &mut Bookkeeping, _copy: &StorageCopy) -> Result<(), IoError> {
            Ok(())
        }

        fn wait_copies(&mut self) -> Result<(), ServerError> {
            Ok(())
        }
    }

    fn reserve(
        arena: &mut PoolArena,
        storage: &mut Bookkeeping,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        arena.reserve(
            storage,
            size,
            PageMapping::Eager,
            &mut ErrorGraph::default(),
        )
    }

    /// Grown to a size the device refused, every page held would be outdated
    /// and every later allocation would need a page of that size too.
    #[test]
    fn a_refused_growth_leaves_the_page_size_where_it_was() {
        let mut storage = Bookkeeping { limit: 8 * MIB };
        let mut arena = PoolArena::new(sizing(), 2);
        let _small = reserve(&mut arena, &mut storage, MIB).unwrap();

        let refused = reserve(&mut arena, &mut storage, 10 * MIB);
        assert!(
            matches!(refused, Err(IoError::OutOfMemory { .. })),
            "{refused:?}"
        );
        assert_eq!(arena.current().page_size(), 2 * MIB);
        assert!(!arena.has_outdated());

        let _served = reserve(&mut arena, &mut storage, MIB).unwrap();
        assert_eq!(arena.current().pages_held(), 1, "served from the page held");
    }

    /// A growth past the last free slot fails on its own terms rather than
    /// overwriting a slot, and a relocation allowed to allocate empties the
    /// outdated pools and frees them.
    #[test]
    fn a_full_arena_refuses_to_grow_until_a_relocation_frees_it() {
        let mut storage = Bookkeeping { limit: u64::MAX };
        let sizing = PageSizing::new(2 * MIB, &MemoryDeviceProperties::new(1024 * MIB, 32));
        let mut arena = PoolArena::new(sizing, 2);
        let failures = &mut ErrorGraph::default();

        // Each size outgrows the page the last one grew to, so every
        // reservation takes a slot, and each pool it leaves keeps one live
        // allocation. The first pool never carved a page.
        let mut live = Vec::new();
        for step in 0..ARENA_SLOTS as u64 - 1 {
            live.push(reserve(&mut arena, &mut storage, (3 + 2 * step) * MIB).unwrap());
        }
        assert!(arena.is_full());
        let refused = reserve(&mut arena, &mut storage, 200 * MIB);
        assert!(
            matches!(refused, Err(IoError::PageSizesExhausted { .. })),
            "{refused:?}"
        );

        let relocation = arena.plan(TargetRoom::MayAllocate, &mut storage, failures);
        assert_eq!(
            relocation.len(),
            ARENA_SLOTS - 2,
            "one allocation per outdated pool that carved a page"
        );
        let landed = relocation.copy(&mut storage, &mut Landing).unwrap();
        arena.commit(landed, &mut storage, failures);

        assert!(!arena.has_outdated());
        reserve(&mut arena, &mut storage, 200 * MIB).expect("a slot is free again");
    }

    #[test]
    fn an_allocation_that_fits_the_pages_grows_nothing() {
        assert_eq!(sizing().fit(MIB, 2 * MIB), Fit::Fits);
    }

    #[test]
    fn a_growth_sizes_pages_to_the_allocation_with_slack() {
        assert_eq!(
            sizing().fit(10 * MIB, 2 * MIB),
            Fit::Grow {
                page_size: 11 * MIB
            }
        );
    }

    #[test]
    fn a_growth_stops_at_the_devices_largest_page() {
        assert_eq!(
            sizing().fit(16 * MIB, 2 * MIB),
            Fit::Grow {
                page_size: 16 * MIB
            }
        );
        assert_eq!(sizing().fit(16 * MIB + 1, 2 * MIB), Fit::TooLarge);
    }
}
