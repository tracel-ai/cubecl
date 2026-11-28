//! Dynamic memory whose pages follow the workload's largest allocation.

use super::{ARENA_SLOT_COUNT, ArenaReserver, PageSizing};
use crate::memory_management::{AllocationKind, Cleanup};
use crate::{
    config::memory::MemoryLogLevel,
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, IMPORTED_POOL_POS, ManagedMemoryBinding, ManagedMemoryHandle, MemoryPoolReport,
        memory_pool::{
            ExclusiveLayout, ExclusiveSliceReserver, PageMapping, PagedLayout, PagedSliceReserver,
            PinnedRoom, SliceBinder, SliceReserver,
        },
        relocation::{Move, RelocationCopier, RelocationLander, RelocationNeed, RelocationReason},
    },
    server::{IoError, ServerError},
    storage::StorageAllocator,
};
use alloc::{format, string::String, vec::Vec};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

/// The pool index zero-sized allocations carry.
const TINY_POOL: u8 = 0;
/// The pool index small allocations carry.
const SMALL_POOL: u8 = 1;
/// The pool index of the arena's first slot.
pub const ARENA_POOLS: u8 = 2;
/// The pool index of the separate arena's first slot, past the arena's.
pub const SEPARATE_POOLS: u8 = ARENA_POOLS + ARENA_SLOT_COUNT as u8;
/// The pool index of the small allocations kept apart, past the separate arena's.
pub const SMALL_SEPARATE_POOL: u8 = SEPARATE_POOLS + ARENA_SLOT_COUNT as u8;

// The arenas' pools are addressed from `ARENA_POOLS` on, one after the other,
// and must stay clear of the fixed indices the persistent, dedicated and
// imported pools carry.
const _: () = assert!(
    (SMALL_SEPARATE_POOL as usize) < IMPORTED_POOL_POS as usize,
    "the arenas' pool indices would reach the fixed ones"
);

/// The small-allocation pool's page size.
const SMALL_PAGE_BYTES: u64 = 8 * 1024 * 1024;
/// The largest allocation the small-allocation pool serves.
const SMALL_SLICE_BYTES: u64 = 64 * 1024;
/// The smallest page the arena carves, capped by the device's
/// `max_page_size`: what the smallest allocation it serves (just past
/// [`SMALL_SLICE_BYTES`]) needs anyway once rounded, so a stream that only makes
/// small allocations holds a page its size rather than a floor's.
const MIN_PAGE_BYTES: u64 = 2 * 1024 * 1024;

/// Reserves allocations from pages sized to the largest allocation served,
/// with a pool per size.
///
/// What a workload allocates decides which pool serves it: zero-sized
/// allocations and the metadata churn have a pool each, and everything else
/// is carved from the [arena](ArenaReserver), whose pages follow the largest
/// allocation served so far.
///
/// `failures` is, in every function, the [`ErrorGraph`] that holds the
/// failures the memory's slices carry.
pub struct AdaptiveReserver {
    /// Zero-sized allocations, which cannot take an offset into a page (on
    /// wgpu at least).
    tiny: ExclusiveSliceReserver,
    /// Kernel metadata — shapes, strides, scalars — churns thousands of tiny
    /// slices. Kept off the arena's pages so they neither fragment them nor
    /// count toward their size.
    small: PagedSliceReserver,
    /// The small allocations kept apart from the room the stream's graphs share
    /// on `small`, as the separate arena keeps the larger ones apart from the
    /// arena's: the metadata a graph keeps, and small work outside a capture
    /// once graphs pin every page of `small`.
    small_separate: PagedSliceReserver,
    /// Everything else.
    arena: ArenaReserver,
    /// What outlives a graph capture — its outputs, the metadata its graph
    /// keeps — apart from the arena, whose pages the stream's graphs share
    /// (see [`AllocationKind::GraphOutput`]), and the work outside a capture that
    /// finds no unpinned room on the arena once graphs pin it. No graph's
    /// replay writes its free room, so they pack together whatever pages
    /// graphs pinned. Never relocated: what it holds is pinned by the graphs
    /// that read and write it.
    separate: ArenaReserver,
    /// What the device holds in total, where it says.
    capacity: Option<u64>,
    logger: Arc<ServerLogger>,
    name: String,
}

impl AdaptiveReserver {
    /// The pools for a device with `properties`.
    pub fn new(
        properties: &MemoryDeviceProperties,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        let small_layout = |pool| PagedLayout {
            page_size_bytes: SMALL_PAGE_BYTES.next_multiple_of(properties.alignment),
            max_slice_bytes: SMALL_SLICE_BYTES.next_multiple_of(properties.alignment),
            alignment_bytes: properties.alignment,
            pool,
        };
        let memory = Self {
            tiny: ExclusiveSliceReserver::new(ExclusiveLayout {
                max_alloc_bytes: 0,
                alignment_bytes: properties.alignment,
                dealloc_period_count: u64::MAX,
                pool: TINY_POOL,
            }),
            // Allocations near its page size belong to the arena, whose
            // pages are sized to what they serve.
            small: PagedSliceReserver::new(small_layout(SMALL_POOL)).up_to_max_slice(),
            small_separate: PagedSliceReserver::new(small_layout(SMALL_SEPARATE_POOL))
                .up_to_max_slice(),
            arena: ArenaReserver::new(
                PageSizing::new(MIN_PAGE_BYTES, properties),
                ARENA_POOLS,
                logger.clone(),
                name.clone(),
            ),
            separate: ArenaReserver::new(
                PageSizing::new(MIN_PAGE_BYTES, properties),
                SEPARATE_POOLS,
                logger.clone(),
                name.clone(),
            ),
            capacity: properties.max_memory(),
            logger,
            name,
        };
        memory.log_layout();
        memory
    }

    /// The pool `pool_index` names, while one is there.
    pub fn pool_by_index(&self, pool_index: u8) -> Option<&dyn SliceReserver> {
        match pool_index {
            TINY_POOL => Some(&self.tiny),
            SMALL_POOL => Some(&self.small),
            SMALL_SEPARATE_POOL => Some(&self.small_separate),
            ARENA_POOLS..SEPARATE_POOLS => Some(self.arena.pool_by_index(pool_index)?),
            _ => Some(self.separate.pool_by_index(pool_index)?),
        }
    }

    /// The pool `pool_index` names, mutably.
    pub fn pool_by_index_mut(&mut self, pool_index: u8) -> Option<&mut dyn SliceReserver> {
        match pool_index {
            TINY_POOL => Some(&mut self.tiny),
            SMALL_POOL => Some(&mut self.small),
            SMALL_SEPARATE_POOL => Some(&mut self.small_separate),
            ARENA_POOLS..SEPARATE_POOLS => Some(self.arena.pool_by_index_mut(pool_index)?),
            _ => Some(self.separate.pool_by_index_mut(pool_index)?),
        }
    }

    /// Install real backing behind `binding` when its allocation was carved
    /// lazily.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the backing.
    pub fn materialize<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let index = binding.descriptor().location().pool;
        match index {
            TINY_POOL => self.tiny.materialize(storage, binding),
            SMALL_POOL => self.small.materialize(storage, binding),
            SMALL_SEPARATE_POOL => self.small_separate.materialize(storage, binding),
            ARENA_POOLS..SEPARATE_POOLS => self.arena.materialize(storage, binding),
            _ => self.separate.materialize(storage, binding),
        }
    }

    /// Reserve `size` bytes on the pool that serves them, growing the arena's
    /// pages when `size` outgrows them.
    ///
    /// A capture's temporaries share the room on pinned pages of the arena or
    /// of the small pool, by size. What [`AllocationKind::GraphOutput`] asks
    /// for, the metadata a capture's info cache keeps for every graph among it,
    /// goes to the separate arena or the small separate pool, since a buffer
    /// kept past its capture must never sit where another graph's replay
    /// writes.
    ///
    /// What [`AllocationKind::Eager`] asks for takes the unpinned room of the
    /// arena or the small pool, and the room kept apart once graphs pin every
    /// page of the small pool, or of the arena's current pool when the pages
    /// of both arenas fit it. A new page of the current size would hold little
    /// more than that buffer: the arena's pages are sized to the largest
    /// temporary, and a recording's input, kept for its graph's life and read
    /// by it, would leave the page pinned and never empty, a page per graph.
    /// One that outgrows the arena's pages grows them as it would.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no page the device allocates fits it,
    /// [`IoError::PageSizesExhausted`] when the arena has no slot left for a
    /// new page size, and whatever the device refused.
    pub fn reserve<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        if self.tiny.accepts_bytes(size) {
            return self.tiny.reserve(storage, size, mapping, failures);
        }
        let Placement { pool, room } = self.placement(size, kind);
        match pool {
            PlacementPool::Small => self
                .small
                .reserve_sharing(storage, size, mapping, room, failures),
            PlacementPool::SmallSeparate => self
                .small_separate
                .reserve_sharing(storage, size, mapping, room, failures),
            PlacementPool::Arena => self.arena.reserve(storage, size, mapping, room, failures),
            PlacementPool::Separate => self
                .separate
                .reserve(storage, size, mapping, room, failures),
        }
    }

    /// Reserve `size` bytes in the room the pools already hold, without
    /// growing the pages, on the pool [`reserve`](Self::reserve) would place
    /// them on. `None` when it has no room for them.
    pub fn try_reserve(
        &mut self,
        size: u64,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        if self.tiny.accepts_bytes(size) {
            return self.tiny.try_reserve(size, failures);
        }
        let Placement { pool, room } = self.placement(size, kind);
        match pool {
            PlacementPool::Small => self.small.try_reserve_sharing(size, room, failures),
            PlacementPool::SmallSeparate => self
                .small_separate
                .try_reserve_sharing(size, room, failures),
            PlacementPool::Arena => self.arena.try_reserve(size, room, failures),
            PlacementPool::Separate => self.separate.try_reserve(size, room, failures),
        }
    }

    /// Whether emptying the arena's outdated pools is wanted (see
    /// [`ArenaReserver::relocation_need`]).
    pub fn relocation_need(&self) -> RelocationNeed {
        self.arena.relocation_need(self.capacity)
    }

    /// Empty the arena's outdated pools into its current one, so the pages
    /// they held go back to the driver (see [`ArenaReserver::relocate`]).
    pub fn relocate<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        copier: &mut dyn RelocationCopier<Storage>,
        reason: RelocationReason,
        failures: &mut ErrorGraph,
    ) {
        self.arena.relocate(storage, copier, reason, failures);
    }

    /// Move every allocation of `kept` that sits on a page a graph guards to the pools kept
    /// apart, with its bytes. Nothing recorded its address, so it may leave a page whose other
    /// slices keep theirs; kept apart, no replay writes the room around it.
    ///
    /// # Guarantees
    ///
    /// An allocation `kept` names more than once moves once.
    ///
    /// # Errors
    ///
    /// When the pools kept apart have no room for one, or a copy fails: nothing moved then.
    pub fn move_apart<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        kept: &[ManagedMemoryBinding],
        copier: &mut dyn RelocationCopier<Storage>,
        failures: &mut ErrorGraph,
    ) -> Result<(), ServerError> {
        let moves = kept
            .iter()
            .map(|memory| self.plan_move_apart(storage, memory, failures))
            .filter_map(Result::transpose)
            .collect::<Result<Vec<_>, _>>()?;
        // SAFETY: each move's target is a fresh reservation on a pool kept
        // apart, of the moved allocation's size, padded like every slice of
        // the pools to the device's alignment, held by nobody but its move;
        // its source is a live allocation on a shared pool, a distinct page.
        let lander = unsafe { RelocationLander::new(moves) };
        let landed = lander.land(storage, copier)?;
        for relocated in landed.into_moves() {
            let (source, destination) = relocated.locations();
            // Only the slices hold the handles once the move goes.
            drop(relocated);
            // A second move of the same allocation finds it apart already, off
            // the shared pools: its target is freed with it.
            if !is_shared_pool_index(source.pool) {
                continue;
            }
            let from = Self::shared_pool(&mut self.small, &mut self.arena, source.pool)
                .expect("a move apart's source should be on a shared pool");
            let into = Self::apart_pool(
                &mut self.small_separate,
                &mut self.separate,
                destination.pool,
            )
            .expect("a move apart's target should be on a pool kept apart");
            SliceBinder.hand_over(from.slice_at(source), into.slice_at(destination), failures);
        }
        Ok(())
    }

    /// Release what the pools no longer need, and drop every outdated pool
    /// that has drained.
    pub fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        reservation_count: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        self.tiny
            .cleanup(storage, reservation_count, cleanup, failures);
        self.small
            .cleanup(storage, reservation_count, cleanup, failures);
        self.small_separate
            .cleanup(storage, reservation_count, cleanup, failures);
        self.arena
            .cleanup(storage, reservation_count, cleanup, failures);
        self.separate
            .cleanup(storage, reservation_count, cleanup, failures);
    }

    /// A report per pool held, in the order allocations are routed through
    /// them, the outdated ones last. The arena's current pool reports how many
    /// pages the outdated ones hold.
    pub fn report(&self) -> Vec<MemoryPoolReport> {
        let mut reports = alloc::vec![
            self.tiny.report(),
            self.small.view().report(),
            self.small_separate.view().report(),
        ];
        self.arena.extend_report(&mut reports);
        self.separate.extend_report(&mut reports);
        reports
    }
}

impl AdaptiveReserver {
    /// Where an allocation of `size` bytes and of `kind` goes, past the tiny pool.
    ///
    /// What carves the pools kept apart is never written by a replay, so their
    /// pinned pages are its to share. Work outside a capture goes there only
    /// once graphs pin every page it could take otherwise: before that, the
    /// unpinned room and a new page of the shared pools serve it.
    fn placement(&self, size: u64, kind: AllocationKind) -> Placement {
        let small = self.small.accepts_bytes(size);
        let (pool, room) = match (kind, small) {
            (AllocationKind::GraphIntermediate, true) => (PlacementPool::Small, PinnedRoom::Share),
            (AllocationKind::GraphIntermediate, false) => (PlacementPool::Arena, PinnedRoom::Share),
            (AllocationKind::GraphOutput, true) => {
                (PlacementPool::SmallSeparate, PinnedRoom::Share)
            }
            (AllocationKind::GraphOutput, false) => (PlacementPool::Separate, PinnedRoom::Share),
            (AllocationKind::Eager, true) if self.small.view().is_pinned() => {
                (PlacementPool::SmallSeparate, PinnedRoom::Share)
            }
            (AllocationKind::Eager, true) => (PlacementPool::Small, PinnedRoom::Avoid),
            (AllocationKind::Eager, false) if self.eager_bytes_go_apart(size) => {
                (PlacementPool::Separate, PinnedRoom::Share)
            }
            (AllocationKind::Eager, false) => (PlacementPool::Arena, PinnedRoom::Avoid),
        };
        Placement { pool, room }
    }

    /// Whether work outside a capture of `size_bytes`, past the small pool, goes to the
    /// separate arena: once graphs pin every page of the arena's current pool, when both
    /// arenas' pages fit it. It never grows the separate arena's pages, whose slots nothing
    /// frees, since it is never relocated: a page size eager work took would be one graphs'
    /// outputs could not.
    fn eager_bytes_go_apart(&self, size_bytes: u64) -> bool {
        self.arena.is_pinned()
            && self.arena.fits_bytes(size_bytes)
            && self.separate.fits_bytes(size_bytes)
    }

    /// The move of `memory` to the pools kept apart, when it sits on a page of the small pool
    /// or of the arena a graph guards: `None` anywhere else, where no replay writes the room
    /// around it.
    fn plan_move_apart<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        memory: &ManagedMemoryBinding,
        failures: &mut ErrorGraph,
    ) -> Result<Option<Move>, IoError> {
        let source = memory.descriptor().location();
        let Some(pool) = Self::shared_pool(&mut self.small, &mut self.arena, source.pool) else {
            return Ok(None);
        };
        let page = pool.page_at(source);
        if !page.view().is_guarded {
            return Ok(None);
        }
        let source_storage = page.view().mapped_storage_at_index(source.slice as usize);
        let slice = pool.slice_at(source);
        let (size, allocation) = (slice.storage.size(), slice.handle.to_binding());
        // Placed as an output: the small pool's slices go to the small pool kept apart, the
        // arena's to the separate arena.
        let target = self.reserve(
            storage,
            size,
            PageMapping::Eager,
            AllocationKind::GraphOutput,
            failures,
        )?;
        let location = target.descriptor().location();
        let copy = Self::apart_pool(&mut self.small_separate, &mut self.separate, location.pool)
            .expect("an output should be placed on a pool kept apart")
            .copy_into(storage, source_storage, location)?;
        Ok(Some(Move {
            allocation,
            target,
            copy,
        }))
    }

    /// The pool of the stream's shared pages, the small pool or one of the arena's, that the
    /// pool `index` names.
    fn shared_pool<'a>(
        small: &'a mut PagedSliceReserver,
        arena: &'a mut ArenaReserver,
        index: u8,
    ) -> Option<&'a mut PagedSliceReserver> {
        match index {
            SMALL_POOL => Some(small),
            ARENA_POOLS..SEPARATE_POOLS => arena.pool_by_index_mut(index),
            _ => None,
        }
    }

    /// The pool kept apart, the small one or one of the separate arena's, that the pool
    /// `index` names.
    fn apart_pool<'a>(
        small_separate: &'a mut PagedSliceReserver,
        separate: &'a mut ArenaReserver,
        index: u8,
    ) -> Option<&'a mut PagedSliceReserver> {
        match index {
            SMALL_SEPARATE_POOL => Some(small_separate),
            SEPARATE_POOLS..SMALL_SEPARATE_POOL => separate.pool_by_index_mut(index),
            _ => None,
        }
    }

    fn log_layout(&self) {
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || format!("[{}] Using memory pools:\n{self}", self.name),
        );
    }
}

/// Whether the pool `index` names is one of the stream's shared pools, the
/// small pool or one of the arena's.
fn is_shared_pool_index(pool_index: u8) -> bool {
    matches!(pool_index, SMALL_POOL | ARENA_POOLS..SEPARATE_POOLS)
}

/// The pool an allocation goes to, and whether it may carve that pool's pinned room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    pub pool: PlacementPool,
    pub room: PinnedRoom,
}

/// The pools [`AdaptiveReserver::placement`] chooses among.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlacementPool {
    Small,
    SmallSeparate,
    Arena,
    Separate,
}

impl core::fmt::Display for AdaptiveReserver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}{}{}{}{}",
            self.tiny, self.small, self.small_separate, self.arena, self.separate
        )
    }
}

// The process mode is never read here: every reservation names its mapping,
// so these are Small Tests.
#[cfg(all(test, not(exclusive_memory_only)))]
mod tests {
    use super::*;
    use crate::memory_management::relocation::HostRelocationCopier;
    use crate::storage::BytesStorage;

    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;

    fn adaptive() -> AdaptiveReserver {
        AdaptiveReserver::new(
            &MemoryDeviceProperties::new(1024 * MIB, 32),
            Arc::new(ServerLogger::default()),
            String::new(),
        )
    }

    fn reserve(
        memory: &mut AdaptiveReserver,
        storage: &mut BytesStorage,
        size: u64,
        kind: AllocationKind,
    ) -> ManagedMemoryHandle {
        memory
            .reserve(
                storage,
                size,
                PageMapping::Eager,
                kind,
                &mut ErrorGraph::default(),
            )
            .expect("the memory should serve the reservation")
    }

    /// A zero-sized allocation goes to the tiny pool.
    #[test]
    fn a_zero_sized_allocation_goes_to_the_tiny_pool() {
        let mut storage = BytesStorage::default();
        let mut memory = adaptive();

        let empty = reserve(&mut memory, &mut storage, 0, AllocationKind::Eager);

        assert_eq!(empty.descriptor().location().pool, TINY_POOL);
    }

    /// A freed zero-sized allocation is served again from the tiny pool.
    #[test]
    fn a_freed_zero_sized_allocation_is_held_by_the_tiny_pool() {
        let mut storage = BytesStorage::default();
        let mut memory = adaptive();
        drop(reserve(&mut memory, &mut storage, 0, AllocationKind::Eager));

        let again = memory
            .try_reserve(0, AllocationKind::Eager, &mut ErrorGraph::default())
            .expect("the tiny pool should hold the freed page");

        assert_eq!(again.descriptor().location().pool, TINY_POOL);
    }

    /// An allocation on a page no graph guards stays where it is.
    #[test]
    fn an_unguarded_allocation_does_not_move_apart() {
        let mut storage = BytesStorage::default();
        let mut memory = adaptive();
        let kept = reserve(&mut memory, &mut storage, 4 * KIB, AllocationKind::Eager);
        let before = kept.descriptor().location();

        memory
            .move_apart(
                &mut storage,
                &[kept.clone().binding()],
                &mut HostRelocationCopier,
                &mut ErrorGraph::default(),
            )
            .expect("nothing should move");

        assert_eq!(kept.descriptor().location(), before);
    }

    /// An allocation already kept apart does not move.
    #[test]
    fn an_allocation_kept_apart_does_not_move() {
        let mut storage = BytesStorage::default();
        let mut memory = adaptive();
        let kept = reserve(
            &mut memory,
            &mut storage,
            4 * KIB,
            AllocationKind::GraphOutput,
        );
        let before = kept.descriptor().location();

        memory
            .move_apart(
                &mut storage,
                &[kept.clone().binding()],
                &mut HostRelocationCopier,
                &mut ErrorGraph::default(),
            )
            .expect("nothing should move");

        assert_eq!(kept.descriptor().location(), before);
    }

    /// An allocation `kept` names twice moves apart once: the second target
    /// is freed with its move.
    #[test]
    fn an_allocation_kept_twice_moves_apart_once() {
        let mut storage = BytesStorage::default();
        let mut memory = adaptive();
        let kept = reserve(&mut memory, &mut storage, 4 * KIB, AllocationKind::Eager);
        let location = kept.descriptor().location();
        let _graph = memory
            .pool_by_index_mut(location.pool)
            .and_then(|pool| pool.guard(location))
            .expect("the kept allocation's page should be guarded");

        memory
            .move_apart(
                &mut storage,
                &[kept.clone().binding(), kept.clone().binding()],
                &mut HostRelocationCopier,
                &mut ErrorGraph::default(),
            )
            .expect("the kept allocation should move apart");

        assert_eq!(
            memory.small_separate.view().report().usage.allocation_count,
            1
        );
    }
}
