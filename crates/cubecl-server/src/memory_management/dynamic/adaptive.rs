//! Dynamic memory whose pages follow the workload's largest allocation.

use super::{ARENA_SLOTS, PageSizing, PoolArena};
use crate::memory_management::{AllocationKind, Cleanup};
use crate::{
    config::memory::MemoryLogLevel,
    logging::ServerLogger,
    memory_management::{
        DEDICATED_POOL_POS, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle,
        MemoryPoolReport,
        memory_pool::{
            ExclusiveLayout, ExclusiveMemoryPool, MemoryPool, PageMapping, PinnedRoom,
            SlicedLayout, SlicedPool,
        },
        relocation::{
            CopyQueue, Move, Relocation, RelocationNeed, RelocationReason, RelocationTrigger,
            StorageCopy,
        },
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
pub(super) const ARENA_POOLS: u8 = 2;
/// The pool index of the separate arena's first slot, past the arena's.
pub(super) const SEPARATE_POOLS: u8 = ARENA_POOLS + ARENA_SLOTS as u8;
/// The pool index of the small allocations kept apart, past the separate arena's.
pub(super) const SMALL_SEPARATE_POOL: u8 = SEPARATE_POOLS + ARENA_SLOTS as u8;

// The arenas' pools are addressed from `ARENA_POOLS` on, one after the other,
// and must stay clear of the fixed indices the dedicated and persistent pools
// carry.
const _: () = assert!(
    (SMALL_SEPARATE_POOL as usize) < DEDICATED_POOL_POS as usize,
    "the arenas' pool indices would reach the fixed ones"
);

/// The small-allocation pool's page size.
const SMALL_PAGE: u64 = 8 * 1024 * 1024;
/// The largest allocation the small-allocation pool serves.
const SMALL_SLICE: u64 = 64 * 1024;
/// The smallest page the arena carves, capped by the device's
/// `max_page_size`: what the smallest allocation it serves (just past
/// [`SMALL_SLICE`]) needs anyway once rounded, so a stream that only makes
/// small allocations holds a page its size rather than a floor's.
const MIN_PAGE: u64 = 2 * 1024 * 1024;

/// Pages sized to the largest allocation served, with a pool per size.
///
/// What a workload allocates decides which pool serves it: zero-sized
/// allocations and the metadata churn have a pool each, and everything else
/// is carved from the [arena](PoolArena), whose pages follow the largest
/// allocation served so far.
pub struct AdaptiveMemory {
    /// Zero-sized allocations, which cannot take an offset into a page (on
    /// wgpu at least).
    tiny: ExclusiveMemoryPool,
    /// Kernel metadata — shapes, strides, scalars — churns thousands of tiny
    /// slices. Kept off the arena's pages so they neither fragment them nor
    /// count toward their size.
    small: SlicedPool,
    /// The small allocations kept apart from the room the stream's graphs share
    /// on `small`, as the separate arena keeps the larger ones apart from the
    /// arena's: the metadata a graph keeps, and small work outside a capture
    /// once graphs pin every page of `small`.
    small_separate: SlicedPool,
    /// Everything else.
    arena: PoolArena,
    /// What outlives a graph capture — its outputs, the metadata its graph
    /// keeps — apart from the arena, whose pages the stream's graphs share
    /// (see [`AllocationKind::GraphOutput`]), and the work outside a capture that
    /// finds no unpinned room on the arena once graphs pin it. No graph's
    /// replay writes its free room, so they pack together whatever pages
    /// graphs pinned. Never relocated: what it holds is pinned by the graphs
    /// that read and write it.
    separate: PoolArena,
    /// When emptying the arena's outdated pools is worth its copies.
    trigger: RelocationTrigger,
    logger: Arc<ServerLogger>,
    name: String,
}

impl AdaptiveMemory {
    /// The pools for a device with `properties`.
    pub fn new(
        properties: &MemoryDeviceProperties,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        let alignment = properties.alignment;
        let min_page = MIN_PAGE.min(properties.max_page_size);
        let sizing = PageSizing::new(min_page, properties);
        let memory = Self {
            tiny: ExclusiveMemoryPool::new(ExclusiveLayout {
                max_alloc_size: 0,
                alignment,
                dealloc_period: u64::MAX,
                pool: TINY_POOL,
            }),
            small: SlicedPool::new(SlicedLayout {
                page_size: SMALL_PAGE.next_multiple_of(alignment),
                max_slice_size: SMALL_SLICE.next_multiple_of(alignment),
                alignment,
                pool: SMALL_POOL,
            })
            // Allocations near its page size belong to the arena, whose
            // pages are sized to what they serve.
            .up_to_max_slice(),
            small_separate: SlicedPool::new(SlicedLayout {
                page_size: SMALL_PAGE.next_multiple_of(alignment),
                max_slice_size: SMALL_SLICE.next_multiple_of(alignment),
                alignment,
                pool: SMALL_SEPARATE_POOL,
            })
            .up_to_max_slice(),
            arena: PoolArena::new(sizing, ARENA_POOLS, logger.clone(), name.clone()),
            separate: PoolArena::new(
                PageSizing::new(min_page, properties),
                SEPARATE_POOLS,
                logger.clone(),
                name.clone(),
            ),
            trigger: RelocationTrigger::new(properties.max_memory()),
            logger,
            name,
        };
        memory.log_layout();
        memory
    }

    /// The pool `index` names, while one is there.
    pub fn pool(&self, index: u8) -> Option<&dyn MemoryPool> {
        match index {
            TINY_POOL => Some(&self.tiny),
            SMALL_POOL => Some(&self.small),
            SMALL_SEPARATE_POOL => Some(&self.small_separate),
            ARENA_POOLS..SEPARATE_POOLS => Some(self.arena.pool(index)?),
            _ => Some(self.separate.pool(index)?),
        }
    }

    /// The pool `index` names, mutably.
    pub fn pool_mut(&mut self, index: u8) -> Option<&mut dyn MemoryPool> {
        match index {
            TINY_POOL => Some(&mut self.tiny),
            SMALL_POOL => Some(&mut self.small),
            SMALL_SEPARATE_POOL => Some(&mut self.small_separate),
            ARENA_POOLS..SEPARATE_POOLS => Some(self.arena.pool_mut(index)?),
            _ => Some(self.separate.pool_mut(index)?),
        }
    }

    /// Install real backing behind `binding` when its allocation was carved
    /// lazily.
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
            ARENA_POOLS..SEPARATE_POOLS => match self.arena.pool_mut(index) {
                Some(pool) => pool.materialize(storage, binding),
                None => Ok(()),
            },
            _ => match self.separate.pool_mut(index) {
                Some(pool) => pool.materialize(storage, binding),
                None => Ok(()),
            },
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
        if self.tiny.accept(size) {
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
        if self.tiny.accept(size) {
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

    /// Whether a growth left pages behind for a relocation to empty.
    pub fn has_outdated(&self) -> bool {
        self.arena.has_outdated()
    }

    /// Whether a relocation is wanted, before the bytes the device holds are
    /// known (see [`RelocationTrigger::need`]).
    pub fn relocation_need(&self) -> RelocationNeed {
        // Asked before every allocation: when nothing is outdated, which is
        // almost always, answer without counting any page.
        if !self.has_outdated() {
            return RelocationNeed::Nothing;
        }
        self.trigger.need(&self.arena.state())
    }

    /// Empty the outdated pools into the current one, so the pages they held
    /// go back to the driver. `reason` decides whether a target may take a
    /// new page.
    ///
    /// A plan whose copies fail is abandoned: every allocation stays where it
    /// was. Targets are freed only once the device is known to be done with them.
    pub fn relocate<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        copier: &mut dyn CopyQueue<Storage>,
        reason: RelocationReason,
        failures: &mut ErrorGraph,
    ) {
        let relocation = self.arena.plan(reason.room(), storage, failures);
        if relocation.is_empty() {
            self.trigger.stalled(&self.arena.state());
            return;
        }
        match relocation.copy(storage, copier) {
            Ok(landed) => {
                self.arena.commit(landed, storage, failures);
                self.trigger.moved();
                // The pages it emptied go back now, not at the storage's next
                // flush: a reservation retried after this needs that room.
                storage.flush();
            }
            // No allocation moved; an inconclusive wait keeps the reservations alive.
            // Settled as moving nothing, so a device that keeps refusing the copies
            // is not waited on again until the pools change.
            Err(err) => {
                self.trigger.stalled(&self.arena.state());
                log::warn!("relocating allocations off outdated memory pages abandoned: {err}")
            }
        }
    }

    /// Move every allocation of `kept` that sits on a page a graph guards to the pools kept
    /// apart, with its bytes. Nothing recorded its address, so it may leave a page whose other
    /// slices keep theirs; kept apart, no replay writes the room around it.
    ///
    /// # Errors
    ///
    /// When the pools kept apart have no room for one, or a copy fails: nothing moved then.
    pub fn move_apart<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        kept: &[ManagedMemoryBinding],
        copier: &mut dyn CopyQueue<Storage>,
        failures: &mut ErrorGraph,
    ) -> Result<(), ServerError> {
        let mut moves = Vec::new();
        for memory in kept {
            if let Some(relocated) = self.plan_move_apart(storage, memory, failures)? {
                moves.push(relocated);
            }
        }
        if moves.is_empty() {
            return Ok(());
        }
        let landed = Relocation::new(moves).copy(storage, copier)?;
        for Move {
            allocation, target, ..
        } in landed.into_moves()
        {
            let source = allocation.descriptor().location();
            let destination = target.descriptor().location();
            // Only the slices hold the handles once these go.
            drop(target);
            drop(allocation);
            let (Some(from), Some(into)) = (
                Self::shared_pool(&mut self.small, &mut self.arena, source.pool),
                Self::apart_pool(
                    &mut self.small_separate,
                    &mut self.separate,
                    destination.pool,
                ),
            ) else {
                continue;
            };
            from.slice_at(source)
                .hand_over(into.slice_at(destination), failures);
        }
        Ok(())
    }

    /// Release what the pools no longer need, and drop every outdated pool
    /// that has drained.
    pub fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        self.tiny.cleanup(storage, alloc_nr, cleanup, failures);
        self.small.cleanup(storage, alloc_nr, cleanup, failures);
        self.small_separate
            .cleanup(storage, alloc_nr, cleanup, failures);
        self.arena.cleanup(storage, alloc_nr, cleanup, failures);
        self.separate.cleanup(storage, alloc_nr, cleanup, failures);
    }

    /// A report per pool held, in the order allocations are routed through
    /// them, the outdated ones last. The arena's current pool reports how many
    /// pages the outdated ones hold.
    pub fn report(&self) -> Vec<MemoryPoolReport> {
        [
            self.tiny.report(),
            self.small.report(),
            self.small_separate.report(),
        ]
        .into_iter()
        .chain(self.arena.report())
        .chain(self.separate.report())
        .collect()
    }
}

impl AdaptiveMemory {
    /// Where an allocation of `size` bytes and of `kind` goes, past the tiny pool.
    ///
    /// What carves the pools kept apart is never written by a replay, so their
    /// pinned pages are its to share. Work outside a capture goes there only
    /// once graphs pin every page it could take otherwise: before that, the
    /// unpinned room and a new page of the shared pools serve it.
    fn placement(&self, size: u64, kind: AllocationKind) -> Placement {
        let small = self.small.accept(size);
        let (pool, room) = match (kind, small) {
            (AllocationKind::GraphIntermediate, true) => (PlacementPool::Small, PinnedRoom::Share),
            (AllocationKind::GraphIntermediate, false) => (PlacementPool::Arena, PinnedRoom::Share),
            (AllocationKind::GraphOutput, true) => {
                (PlacementPool::SmallSeparate, PinnedRoom::Share)
            }
            (AllocationKind::GraphOutput, false) => (PlacementPool::Separate, PinnedRoom::Share),
            (AllocationKind::Eager, true) if self.small.is_pinned() => {
                (PlacementPool::SmallSeparate, PinnedRoom::Share)
            }
            (AllocationKind::Eager, true) => (PlacementPool::Small, PinnedRoom::Avoid),
            (AllocationKind::Eager, false) if self.eager_goes_apart(size) => {
                (PlacementPool::Separate, PinnedRoom::Share)
            }
            (AllocationKind::Eager, false) => (PlacementPool::Arena, PinnedRoom::Avoid),
        };
        Placement { pool, room }
    }

    /// Whether work outside a capture of `size` bytes, past the small pool, goes to the separate
    /// arena: once graphs pin every page of the arena's current pool, when both arenas' pages
    /// fit it. It never grows the separate arena's pages, whose slots nothing frees, since it is
    /// never relocated: a page size eager work took would be one graphs' outputs could not.
    fn eager_goes_apart(&self, size: u64) -> bool {
        self.arena.is_pinned() && self.arena.fits(size) && self.separate.fits(size)
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
        if !pool.is_guarded_at(source) {
            return Ok(None);
        }
        let slice = pool.slice_at(source);
        let (size, allocation) = (slice.storage.size(), slice.handle.to_binding());
        let source_storage = pool.storage_at(source);
        // Placed as an output: the small pool's slices go to the small pool kept apart, the
        // arena's to the separate arena.
        let target = self.reserve(
            storage,
            size,
            PageMapping::Eager,
            AllocationKind::GraphOutput,
            failures,
        )?;
        let copy = match source_storage {
            Some(source) => {
                let location = target.descriptor().location();
                let Some(pool) =
                    Self::apart_pool(&mut self.small_separate, &mut self.separate, location.pool)
                else {
                    return Ok(None);
                };
                Some(StorageCopy {
                    source,
                    target: pool.mapped_storage_at(storage, location)?,
                })
            }
            None => None,
        };
        Ok(Some(Move {
            allocation,
            target,
            copy,
        }))
    }

    /// The pool of the stream's shared pages, the small pool or one of the arena's, that the
    /// pool `index` names.
    fn shared_pool<'a>(
        small: &'a mut SlicedPool,
        arena: &'a mut PoolArena,
        index: u8,
    ) -> Option<&'a mut SlicedPool> {
        match index {
            SMALL_POOL => Some(small),
            ARENA_POOLS..SEPARATE_POOLS => arena.pool_mut(index),
            _ => None,
        }
    }

    /// The pool kept apart, the small one or one of the separate arena's, that the pool
    /// `index` names.
    fn apart_pool<'a>(
        small_separate: &'a mut SlicedPool,
        separate: &'a mut PoolArena,
        index: u8,
    ) -> Option<&'a mut SlicedPool> {
        match index {
            SMALL_SEPARATE_POOL => Some(small_separate),
            SEPARATE_POOLS..SMALL_SEPARATE_POOL => separate.pool_mut(index),
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

/// The pool an allocation goes to, and whether it may carve that pool's pinned room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    pub pool: PlacementPool,
    pub room: PinnedRoom,
}

/// The pools [`AdaptiveMemory::placement`] chooses among.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlacementPool {
    Small,
    SmallSeparate,
    Arena,
    Separate,
}

impl core::fmt::Display for AdaptiveMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}{}{}{}{}",
            self.tiny, self.small, self.small_separate, self.arena, self.separate
        )
    }
}
