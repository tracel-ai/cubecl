//! The pools a workload's allocations are carved from, one per page size it
//! grew through.

use super::{Fit, PageSizing};
use crate::memory_management::Cleanup;
use crate::{
    config::memory::MemoryLogLevel,
    logging::ServerLogger,
    memory_management::{
        BytesFormat, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryPoolKind,
        MemoryPoolReport,
        memory_pool::{
            PageMapping, PageSuballocator, PagedSliceReserver, PinnedRoom, SliceBinder,
            SliceReserver,
        },
        relocation::{
            ArenaShape, ArenaState, LandedRelocation, LiveAllocation, MovablePage, Move,
            OutdatedPages, RelocationCopier, RelocationDecider, RelocationLander, RelocationNeed,
            RelocationReason, TargetRoom,
        },
    },
    server::{IoError, ServerError},
    storage::StorageAllocator,
};
use alloc::{format, string::String, vec::Vec};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::sync::Arc;

/// How many page sizes the arena holds at once.
///
/// A slot frees once the pool in it drains, and a relocation drains the
/// outdated ones, so a workload reaches this only by growing through that many
/// sizes while each keeps something alive.
pub(crate) const ARENA_SLOT_COUNT: usize = 64;

/// Reserves allocations from a fixed number of paged pools, one per page size
/// a workload's allocations grew through.
///
/// The *current* pool carves every new allocation. When one outgrows its
/// pages, a pool of the new size takes over and the old one is *outdated*: it
/// serves no new reservation, gives each page back as it empties, and frees
/// its slot once it holds none.
///
/// `failures` is, in every function, the [`ErrorGraph`] that holds the
/// failures the arena's slices carry.
pub(crate) struct ArenaReserver {
    /// The pools by slot (Position); a slot is empty once its pool drained.
    slots: Vec<Option<PagedSliceReserver>>,
    /// The pool index a slice's location carries for slot 0.
    first_index: u8,
    /// The slot new allocations are carved from.
    current: usize,
    /// The size the current pool's pages are allocated at.
    current_page_size: u64,
    /// The slots a growth left behind, in the order they were outdated
    /// (Arrival), drained as their pools empty.
    outdated: Vec<usize>,
    sizing: PageSizing,
    /// When emptying the outdated pools is worth its copies.
    decider: RelocationDecider,
    logger: Arc<ServerLogger>,
    /// The memory the arena belongs to, as its logs name it.
    name: String,
}

impl ArenaReserver {
    /// An arena whose first pool carves the smallest pages `sizing` allows,
    /// its slots addressed from `first_index` on.
    pub(crate) fn new(
        sizing: PageSizing,
        first_index: u8,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        let mut slots = Vec::with_capacity(ARENA_SLOT_COUNT);
        slots.push(Some(PagedSliceReserver::new(
            sizing.layout(sizing.min_page_bytes, first_index),
        )));
        slots.resize_with(ARENA_SLOT_COUNT, || None);
        Self {
            slots,
            first_index,
            current: 0,
            current_page_size: sizing.min_page_bytes,
            outdated: Vec::with_capacity(ARENA_SLOT_COUNT),
            sizing,
            decider: RelocationDecider::default(),
            logger,
            name,
        }
    }

    /// The pool `pool_index` names, while one is there.
    pub(crate) fn pool_by_index(&self, pool_index: u8) -> Option<&PagedSliceReserver> {
        let slot = pool_index.checked_sub(self.first_index)?;
        self.slots.get(slot as usize)?.as_ref()
    }

    /// The pool `pool_index` names, mutably.
    pub(crate) fn pool_by_index_mut(&mut self, pool_index: u8) -> Option<&mut PagedSliceReserver> {
        let slot = pool_index.checked_sub(self.first_index)?;
        self.slots.get_mut(slot as usize)?.as_mut()
    }

    /// Install real backing behind `binding` when its allocation was carved
    /// lazily on one of the arena's pools; a binding of no pool it holds is
    /// left alone.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the backing.
    pub(crate) fn materialize<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        self.pool_by_index_mut(binding.descriptor().location().pool)
            .map_or(Ok(()), |pool| pool.materialize(storage, binding))
    }

    /// Whether a page of the size the arena carves now holds `size_bytes`: a
    /// new page for them would not outdate the current ones.
    pub(crate) fn fits_bytes(&self, size_bytes: u64) -> bool {
        self.sizing.fit(size_bytes, self.current_page_size) == Fit::Fits
    }

    /// Whether the current pool is pinned (see
    /// [`PagedView::is_pinned`](crate::memory_management::memory_pool::PagedView::is_pinned)).
    pub(crate) fn is_pinned(&self) -> bool {
        self.current().view().is_pinned()
    }

    /// Whether emptying the outdated pools is wanted, on a device that holds
    /// `capacity` bytes where it says: only while something is outdated, when
    /// the arena is full or the device may be short of room, and not again
    /// while the pools look as they did when the last relocation moved
    /// nothing.
    pub(crate) fn relocation_need(&self, capacity: Option<u64>) -> RelocationNeed {
        // Asked before every allocation: when nothing is outdated, which is
        // almost always, answer without counting any page.
        if self.outdated.is_empty() {
            return RelocationNeed::Nothing;
        }
        self.decider.decide(&self.state(), capacity)
    }

    /// Add a report for the current pool, stating how many pages the outdated
    /// ones hold, then one per outdated pool, to `reports`.
    pub(crate) fn extend_report(&self, reports: &mut Vec<MemoryPoolReport>) {
        reports.push(MemoryPoolReport {
            kind: MemoryPoolKind::Adaptive {
                page_size: self.current_page_size,
                outdated_pages: self.outdated_page_count(),
            },
            ..self.current().view().report()
        });
        reports.extend(self.outdated().map(|pool| pool.view().report()));
    }

    /// Reserve `size` bytes on the current pool, growing the pages when `size`
    /// outgrows them.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no page the device allocates fits it,
    /// [`IoError::PageSizesExhausted`] when it needs a new page size and
    /// every slot is taken, and whatever the device refused.
    pub(crate) fn reserve<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        room: PinnedRoom,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match self.sizing.fit(size, self.current_page_size) {
            Fit::Fits => self
                .current_mut()
                .reserve_sharing(storage, size, mapping, room, failures),
            Fit::Grow { page_size_bytes } => {
                self.grow(storage, size, page_size_bytes, mapping, failures)
            }
            Fit::TooLarge => Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// Reserve `size` bytes in the room the current pool holds. `None` when
    /// it has none for it, or when it would need larger pages.
    ///
    /// Only the current pool carves, so the room on a pinned page an outdated
    /// pool holds serves no capture after the growth: why the largest graph
    /// is recorded first.
    pub(crate) fn try_reserve(
        &mut self,
        size: u64,
        room: PinnedRoom,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        if !self.fits_bytes(size) {
            return None;
        }
        self.current_mut().try_reserve_sharing(size, room, failures)
    }

    /// Release what the current pool no longer needs, and drop every outdated
    /// pool that has drained.
    pub(crate) fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        reservation_count: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        self.current_mut()
            .cleanup(storage, reservation_count, cleanup, failures);
        self.drain(storage, failures);
    }

    /// Empty the outdated pools into the current one, so the pages they held
    /// go back to the driver. `reason` decides whether a target may take a
    /// new page.
    ///
    /// A plan whose copies fail is abandoned: every allocation stays where it
    /// was. Targets are freed only once every later use of them runs after the
    /// copies.
    pub(crate) fn relocate<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        copier: &mut dyn RelocationCopier<Storage>,
        reason: RelocationReason,
        failures: &mut ErrorGraph,
    ) {
        if self.outdated.is_empty() {
            return;
        }
        let relocation = self.plan(reason.room(), storage, failures);
        if relocation.is_empty() {
            self.decider.stalled(&self.state());
            return;
        }
        match relocation.land(storage, copier) {
            Ok(landed) => self.land(landed, storage, failures),
            Err(err) => self.abandon(&err),
        }
    }

    /// The pool new allocations are carved from.
    fn current(&self) -> &PagedSliceReserver {
        self.slots[self.current]
            .as_ref()
            .expect("the current slot should always hold a pool")
    }

    /// The pool new allocations are carved from, mutably.
    fn current_mut(&mut self) -> &mut PagedSliceReserver {
        self.slots[self.current]
            .as_mut()
            .expect("the current slot should always hold a pool")
    }

    /// The pools a growth left behind.
    fn outdated(&self) -> impl Iterator<Item = &PagedSliceReserver> {
        self.outdated.iter().map(|slot| {
            self.slots[*slot]
                .as_ref()
                .expect("an outdated slot should hold its pool until it drains")
        })
    }

    /// The pages the outdated pools hold.
    fn outdated_pages(&self) -> impl Iterator<Item = &PageSuballocator> {
        self.outdated().flat_map(|pool| pool.view().pages())
    }

    /// How many pages the outdated pools hold.
    fn outdated_page_count(&self) -> u64 {
        self.outdated_pages().count() as u64
    }

    /// What a relocation decider reads of the arena.
    fn state(&self) -> ArenaState {
        ArenaState {
            page_size_bytes: self.current_page_size,
            // Only the current pool and the outdated ones hold a slot.
            is_full: self.outdated.len() + 1 == ARENA_SLOT_COUNT,
            shape: ArenaShape {
                current_page_count: self.current().view().pages().len() as u64,
                outdated_pool_count: self.outdated.len(),
                outdated_page_count: self.outdated_page_count(),
                outdated_guarded_count: self
                    .outdated_pages()
                    .filter(|page| page.view().is_guarded)
                    .count(),
            },
        }
    }

    /// The pools `source` and `target` name, which are distinct.
    fn pair_mut(
        &mut self,
        source: u8,
        target: u8,
    ) -> (&mut PagedSliceReserver, &mut PagedSliceReserver) {
        let first = self.first_index;
        let slot = |pool: u8| {
            pool.checked_sub(first)
                .expect("a moved allocation should be on one of the arena's pools")
                as usize
        };
        let [source, target] = self
            .slots
            .get_disjoint_mut([slot(source), slot(target)])
            .expect("a move should be between two of the arena's slots");
        (
            source
                .as_mut()
                .expect("a move's source pool should be held until it commits"),
            target
                .as_mut()
                .expect("a move's target pool should be the current one"),
        )
    }

    /// Give every empty page of the outdated pools back, and free the slot of
    /// each pool left with none.
    ///
    /// An outdated pool returns a page the moment it empties, rather than
    /// waiting for the explicit cleanup the current pool waits for.
    fn drain<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        let slots = &mut self.slots;
        self.outdated.retain(|&slot| {
            let pool = slots[slot]
                .as_mut()
                .expect("an outdated slot should hold its pool until it drains");
            pool.release_empty(storage, failures);
            let drained = pool.view().pages().is_empty();
            if drained {
                slots[slot] = None;
            }
            !drained
        });
    }

    /// Reserve a slice on the current pool for every live allocation on the
    /// outdated pages [the analysis](OutdatedPages::into_plan) plans to empty,
    /// so those pools drain instead of waiting on their longest-lived slice.
    ///
    /// `room` says whether a target may take a new page when the room held
    /// runs out. Nothing moves yet: the caller copies the bytes and hands the
    /// [`LandedRelocation`] back to [`land`](Self::land).
    fn plan<Storage: StorageAllocator>(
        &mut self,
        room: TargetRoom,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> RelocationLander {
        let plan = OutdatedPages::new(self.outdated_pages()).into_plan();
        let moves = plan
            .into_iter()
            .filter_map(|page| self.plan_page(page, room, storage, failures))
            .flatten()
            .collect();
        // SAFETY: each move's target is a fresh reservation on the current
        // pool, of the moved allocation's size, padded like every slice of the
        // pool to the device's alignment, held by nobody but its move; its
        // source is a live allocation on an outdated pool, a distinct page.
        unsafe { RelocationLander::new(moves) }
    }

    /// The moves of every allocation of `page`, or `None` when one of them
    /// finds no target: moving part of what is on a page frees nothing, and
    /// the targets reserved for the others are freed with the moves. A page
    /// later in the plan may still fit what this one did not.
    fn plan_page<Storage: StorageAllocator>(
        &mut self,
        page: MovablePage,
        room: TargetRoom,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> Option<Vec<Move>> {
        page.live
            .into_iter()
            .map(|allocation| self.plan_move(allocation, room, storage, failures))
            .collect()
    }

    /// Hand every allocation of `landed` over to the target reserved for it,
    /// now that its bytes are there, drop the outdated pools that emptied,
    /// and give the pages they held back now rather than at the storage's
    /// next flush: a reservation retried after this needs that room.
    fn land<Storage: StorageAllocator>(
        &mut self,
        landed: LandedRelocation,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        for relocated in landed.into_moves() {
            let (source, destination) = relocated.locations();
            // Only the slices hold the handles once the move goes.
            drop(relocated);
            let (source_pool, target_pool) = self.pair_mut(source.pool, destination.pool);
            SliceBinder.hand_over(
                source_pool.slice_at(source),
                target_pool.slice_at(destination),
                failures,
            );
        }
        self.drain(storage, failures);
        self.decider.moved();
        storage.flush();
    }

    /// Settle a relocation the device refused as one that moved nothing, so
    /// a device that keeps refusing the copies is not waited on again until
    /// the pools change. No allocation moved; an inconclusive wait keeps the
    /// reservations alive.
    fn abandon(&mut self, err: &ServerError) {
        self.decider.stalled(&self.state());
        log::warn!("relocating allocations off outdated memory pages abandoned: {err}");
    }

    /// Reserve a target for `allocation` on the current pool, and say what has
    /// to be copied into it. `None` when `room` has no place for it.
    fn plan_move<Storage: StorageAllocator>(
        &mut self,
        allocation: LiveAllocation,
        room: TargetRoom,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> Option<Move> {
        let source = allocation.handle.descriptor().location();
        let source_storage = self
            .pool_by_index(source.pool)
            .expect("an outdated allocation should be on one of the arena's pools")
            .page_at(source)
            .view()
            .mapped_storage_at_index(source.slice as usize);

        let pool = self.current_mut();
        let target = match (pool.try_reserve(allocation.size_bytes, failures), room) {
            (Some(target), _) => target,
            (None, TargetRoom::Held) => return None,
            (None, TargetRoom::MayAllocate) => pool
                .alloc(storage, allocation.size_bytes, PageMapping::Eager, failures)
                .ok()?,
        };
        // Bytes that exist have to land somewhere real; a source never
        // resolved has none to copy.
        let copy = pool
            .copy_into(storage, source_storage, target.descriptor().location())
            .ok()?;

        Some(Move {
            allocation: allocation.handle,
            target,
            copy,
        })
    }

    /// Carve `size` bytes from a new pool of `page_size` pages, which takes
    /// over from the current one only once the device gave it its first page:
    /// a refused allocation leaves the page size where it was.
    fn grow<Storage: StorageAllocator>(
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
        let mut pool =
            PagedSliceReserver::new(self.sizing.layout(page_size, self.first_index + slot as u8));
        let handle = pool.alloc(storage, size, mapping, failures)?;

        self.slots[slot] = Some(pool);
        self.outdated.push(self.current);
        self.current = slot;
        self.current_page_size = page_size;
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                format!(
                    "[{}] Carving pages of {} from here on:\n{self}",
                    self.name,
                    BytesFormat::from_byte_count(page_size)
                )
            },
        );
        Ok(handle)
    }
}

impl core::fmt::Display for ArenaReserver {
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
    use crate::memory_management::relocation::StorageCopy;
    use crate::storage::{StorageHandle, StorageId, StorageUtilization};
    use cubecl_ir::MemoryDeviceProperties;

    const MIB: u64 = 1024 * 1024;

    fn sizing() -> PageSizing {
        PageSizing::new(2 * MIB, &MemoryDeviceProperties::new(16 * MIB, 32))
    }

    /// Device memory that is only bookkeeping, refusing any allocation past
    /// `limit` bytes the way a full device would.
    struct BookkeepingStorageAllocator {
        limit: u64,
    }

    impl StorageAllocator for BookkeepingStorageAllocator {
        type Resource = ();

        fn get(&mut self, _handle: &StorageHandle) {}

        fn allocate_bytes(&mut self, size: u64) -> Result<StorageHandle, IoError> {
            if size > self.limit {
                return Err(IoError::OutOfMemory {
                    size,
                    backtrace: BackTrace::capture(),
                });
            }
            Ok(StorageHandle::new(
                StorageId::new(),
                StorageUtilization {
                    offset_bytes: 0,
                    size_bytes: size,
                },
            ))
        }

        fn deallocate(&mut self, _id: StorageId) {}

        fn flush(&mut self) {}

        fn bytes_allocated(&self) -> u64 {
            0
        }
    }

    /// Copies that land the moment they are made.
    struct LandingRelocationCopier;

    // SAFETY: test memory that no work reads or writes, and copies that do
    // nothing or land on return.
    unsafe impl RelocationCopier<BookkeepingStorageAllocator> for LandingRelocationCopier {
        fn wait_device(&mut self) -> Result<(), ServerError> {
            Ok(())
        }

        unsafe fn copy(
            &mut self,
            _storage: &mut BookkeepingStorageAllocator,
            _copy: &StorageCopy,
        ) -> Result<(), IoError> {
            Ok(())
        }

        fn wait_copies(&mut self) -> Result<(), ServerError> {
            Ok(())
        }
    }

    fn arena(sizing: PageSizing) -> ArenaReserver {
        ArenaReserver::new(sizing, 2, Default::default(), String::new())
    }

    fn reserve(
        arena: &mut ArenaReserver,
        storage: &mut BookkeepingStorageAllocator,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        arena.reserve(
            storage,
            size,
            PageMapping::Eager,
            PinnedRoom::Avoid,
            &mut ErrorGraph::default(),
        )
    }

    /// Grown to a size the device refused, every page held would be outdated
    /// and every later allocation would need a page of that size too.
    #[test]
    fn a_refused_growth_leaves_the_page_size_where_it_was() {
        let mut storage = BookkeepingStorageAllocator { limit: 8 * MIB };
        let mut arena = arena(sizing());
        let _small = reserve(&mut arena, &mut storage, MIB).expect("1 MiB should fit a page");

        let refused = reserve(&mut arena, &mut storage, 10 * MIB);
        assert!(
            matches!(refused, Err(IoError::OutOfMemory { .. })),
            "{refused:?}"
        );
        assert_eq!(arena.current_page_size, 2 * MIB);
        assert!(arena.outdated.is_empty());

        let _served = reserve(&mut arena, &mut storage, MIB).expect("1 MiB should fit a page");
        assert_eq!(
            arena.current().view().pages().len(),
            1,
            "served from the page held"
        );
    }

    /// A growth past the last free slot fails on its own terms rather than
    /// overwriting a slot, and a relocation allowed to allocate empties the
    /// outdated pools and frees them.
    #[test]
    fn a_full_arena_refuses_to_grow_until_a_relocation_frees_it() {
        let mut storage = BookkeepingStorageAllocator { limit: u64::MAX };
        let mut arena = arena(PageSizing::new(
            2 * MIB,
            &MemoryDeviceProperties::new(1024 * MIB, 32),
        ));

        // Each size outgrows the page the last one grew to, so every
        // reservation takes a slot, and each pool it leaves keeps one live
        // allocation. The first pool never carved a page.
        let mut live = Vec::new();
        for step in 0..ARENA_SLOT_COUNT as u64 - 1 {
            live.push(
                reserve(&mut arena, &mut storage, (3 + 2 * step) * MIB)
                    .expect("a slot should be free"),
            );
        }
        assert!(arena.state().is_full);
        let refused = reserve(&mut arena, &mut storage, 200 * MIB);
        assert!(
            matches!(refused, Err(IoError::PageSizesExhausted { .. })),
            "{refused:?}"
        );
        assert_eq!(
            arena.relocation_need(None),
            RelocationNeed::Wanted(RelocationReason::ArenaFull)
        );

        arena.relocate(
            &mut storage,
            &mut LandingRelocationCopier,
            RelocationReason::ArenaFull,
            &mut ErrorGraph::default(),
        );

        assert!(arena.outdated.is_empty());
        reserve(&mut arena, &mut storage, 200 * MIB).expect("a slot should be free again");
    }

    /// With nothing outdated, nothing is wanted, and the bytes are never read.
    #[test]
    fn nothing_outdated_needs_nothing() {
        let mut storage = BookkeepingStorageAllocator { limit: u64::MAX };
        let mut arena = arena(sizing());
        let _small = reserve(&mut arena, &mut storage, MIB).expect("1 MiB should fit a page");

        assert_eq!(arena.relocation_need(Some(1)), RelocationNeed::Nothing);
    }

    /// A reservation that would need larger pages finds no room held.
    #[test]
    fn a_reservation_needing_larger_pages_finds_no_room_held() {
        let mut storage = BookkeepingStorageAllocator { limit: u64::MAX };
        let mut arena = arena(sizing());
        core::mem::drop(reserve(&mut arena, &mut storage, MIB).expect("1 MiB should fit a page"));

        assert!(
            arena
                .try_reserve(4 * MIB, PinnedRoom::Avoid, &mut ErrorGraph::default())
                .is_none()
        );
        assert!(
            arena
                .try_reserve(MIB, PinnedRoom::Avoid, &mut ErrorGraph::default())
                .is_some()
        );
    }

    /// A relocation that finds nothing to move settles, and is not wanted
    /// again until the pools change.
    #[test]
    fn a_relocation_with_nothing_to_move_settles() {
        let mut storage = BookkeepingStorageAllocator { limit: u64::MAX };
        let mut arena = arena(sizing());
        let small = reserve(&mut arena, &mut storage, MIB).expect("1 MiB should fit a page");
        // A binding kept past the work: the page it sits on cannot move.
        let _read_back = small.clone().binding();
        let _large = reserve(&mut arena, &mut storage, 10 * MIB).expect("a slot should be free");
        let capacity = Some(16 * MIB);
        assert!(matches!(
            arena.relocation_need(capacity),
            RelocationNeed::UnderPressure { .. }
        ));

        arena.relocate(
            &mut storage,
            &mut LandingRelocationCopier,
            RelocationReason::MemoryPressure,
            &mut ErrorGraph::default(),
        );

        assert_eq!(arena.relocation_need(capacity), RelocationNeed::Settled);
    }
}
