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
    storage::ComputeStorage,
};
use alloc::{format, string::String, vec::Vec};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

/// The pool index zero-sized allocations carry.
const TINY_POOL: u8 = 0;
/// The pool index small allocations carry.
const SMALL_POOL: u8 = 1;
/// The pool index of the arena's first slot.
const ARENA_POOLS: u8 = 2;
/// The pool index of the separate arena's first slot, past the arena's.
const SEPARATE_POOLS: u8 = ARENA_POOLS + ARENA_SLOTS as u8;
/// The pool index of the small allocations kept apart, past the separate arena's.
const SMALL_SEPARATE_POOL: u8 = SEPARATE_POOLS + ARENA_SLOTS as u8;

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
    pub fn materialize<Storage: ComputeStorage>(
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
    /// of both arenas fit it. A new page of the current size would hold little more than
    /// that buffer: the arena's pages are sized to the largest temporary, and a
    /// recording's input, kept for its graph's life and read by it, would leave
    /// the page pinned and never empty, a page per graph. One that outgrows the
    /// arena's pages grows them as it would.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no page the device allocates fits it,
    /// [`IoError::PageSizesExhausted`] when the arena has no slot left for a
    /// new page size, and whatever the device refused.
    pub fn reserve<Storage: ComputeStorage>(
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
    pub fn relocate<Storage: ComputeStorage>(
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
    pub fn move_apart<Storage: ComputeStorage>(
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
    pub fn cleanup<Storage: ComputeStorage>(
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
    fn plan_move_apart<Storage: ComputeStorage>(
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
        // Safety: the binding runs on the device thread in the stream's order, ahead of any later
        // launch of the buffer's owner.
        let (size, allocation) = (slice.storage.size(), unsafe {
            slice.handle.uncounted_binding()
        });
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

// The layout these tests drive is the `Adaptive` preset, which a build
// refusing sub-slicing does not have.
//
// Serial with the tests that open a process mode override: one that discards
// launches maps pages lazily, so the pages these tests reserve would hold no
// device bytes and the memory pressure they measure would read as none.
#[cfg(all(test, not(exclusive_memory_only)))]
#[serial_test::serial(records)]
mod tests {
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use crate::{
        logging::ServerLogger,
        memory_management::{
            AllocationKind, ErrorGraph, ManagedMemoryHandle, MemoryAllocationMode,
            MemoryConfiguration, MemoryManagement, MemoryManagementOptions, MemoryPoolKind,
            PageUpdate,
            relocation::{CopyQueue, HostCopies, RelocationReason, StorageCopy},
        },
        server::{IoError, ServerError},
        storage::BytesStorage,
    };
    use alloc::vec;
    use cubecl_environment::sync::Arc;
    use cubecl_ir::MemoryDeviceProperties;

    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const PROPERTIES: MemoryDeviceProperties = MemoryDeviceProperties::new(1024 * MIB, 32);

    /// The floor the `Adaptive` preset carves its pages at, until a larger
    /// allocation grows them.
    const FLOOR: u64 = 2 * MIB;

    /// A memory management laid out by the `Adaptive` preset, on a device
    /// that holds `max_memory` bytes.
    fn adaptive_on_device(max_memory: u64) -> MemoryManagement<BytesStorage> {
        MemoryManagement::from_configuration(
            BytesStorage::default(),
            &PROPERTIES.with_max_memory(max_memory),
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            MemoryManagementOptions::new("adaptive"),
        )
    }

    /// A memory management laid out by the `Adaptive` preset. Every size these
    /// tests reserve is past the metadata pool's largest slice, so the arena
    /// is what serves them.
    fn adaptive() -> MemoryManagement<BytesStorage> {
        MemoryManagement::from_configuration(
            BytesStorage::default(),
            &PROPERTIES,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            MemoryManagementOptions::new("adaptive"),
        )
    }

    /// The adaptive pool, as its report states it.
    #[derive(Debug, PartialEq, Eq)]
    struct Pool {
        page_size: u64,
        pages: u64,
        outdated: u64,
    }

    /// The adaptive pool: the size it carves pages at, every page it holds —
    /// outdated ones included — and how many of those are outdated.
    fn pool(memory: &MemoryManagement<BytesStorage>) -> Pool {
        let report = memory.memory_report().dynamic;
        let current = report
            .iter()
            .find_map(|pool| match pool.kind {
                MemoryPoolKind::Adaptive {
                    page_size,
                    outdated_pages,
                } => Some((page_size, outdated_pages)),
                _ => None,
            })
            .expect("the layer carves allocations from adaptive pages");
        Pool {
            page_size: current.0,
            pages: report.iter().map(|pool| pool.pages).sum(),
            outdated: current.1,
        }
    }

    fn reserve(memory: &mut MemoryManagement<BytesStorage>, size: u64) -> ManagedMemoryHandle {
        memory
            .reserve(size, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap()
    }

    /// Where an allocation sits: its pool, page and slice.
    fn place(handle: &ManagedMemoryHandle) -> (u8, u16, u32) {
        let location = handle.descriptor().location();
        (location.pool, location.page, location.slice)
    }

    /// The page an allocation sits on, which only its pool numbers.
    fn page_of(handle: &ManagedMemoryHandle) -> (u8, u16) {
        let location = handle.descriptor().location();
        (location.pool, location.page)
    }

    fn fill(memory: &mut MemoryManagement<BytesStorage>, handle: &ManagedMemoryHandle, byte: u8) {
        let mut resource = memory
            .get_resource(handle.clone().binding(), None, None)
            .unwrap();
        resource.write().fill(byte);
    }

    fn contents(memory: &mut MemoryManagement<BytesStorage>, handle: &ManagedMemoryHandle) -> u8 {
        let resource = memory
            .get_resource(handle.clone().binding(), None, None)
            .unwrap();
        let bytes = resource.read();
        assert!(bytes.iter().all(|byte| *byte == bytes[0]));
        bytes[0]
    }

    /// Run a relocation the way a command does, with a host copy standing in
    /// for the device one. Answers how many allocations moved.
    fn relocate(memory: &mut MemoryManagement<BytesStorage>) -> usize {
        let failures = &mut ErrorGraph::default();
        let before = memory.memory_report().dynamic.len();
        let mut copies = HostCopies;
        memory.relocate(&mut copies, RelocationReason::Explicit, failures);
        // A pool a relocation emptied is dropped, so the reports it leaves say
        // how many moved off it.
        before - memory.memory_report().dynamic.len()
    }

    /// A device that refuses every copy, which abandons the relocation.
    struct RefusedCopies;

    impl CopyQueue<BytesStorage> for RefusedCopies {
        fn wait_device(&mut self) -> Result<(), ServerError> {
            Ok(())
        }

        fn copy(
            &mut self,
            _storage: &mut BytesStorage,
            _copy: &StorageCopy,
        ) -> Result<(), IoError> {
            Err(IoError::Unknown {
                description: "refused".into(),
                backtrace: Default::default(),
            })
        }

        fn wait_copies(&mut self) -> Result<(), ServerError> {
            Ok(())
        }
    }

    /// Whether a relocation is wanted, as the server asks before a
    /// reservation, with this memory the only one on the device.
    fn relocation(memory: &MemoryManagement<BytesStorage>) -> Option<RelocationReason> {
        memory.relocation_need().reason(|| memory.bytes_allocated())
    }

    /// A device with room to spare says nothing; one whose next page would
    /// leave it less than another page asks for the outdated pools to be
    /// emptied first, which is the last moment a relocation has somewhere to
    /// copy to.
    #[test]
    fn memory_pressure_asks_for_a_relocation_before_the_next_page() {
        const MAX_MEMORY: u64 = 24 * MIB;
        let mut memory = adaptive_on_device(MAX_MEMORY);

        let kept = reserve(&mut memory, MIB);
        assert_eq!(relocation(&memory), None, "one 2 MiB page of 24 MiB");

        // 13 MiB held over two pools, and the next page is 11 MiB.
        let large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocation(&memory), Some(RelocationReason::MemoryPressure));

        let mut copies = HostCopies;
        memory.relocate(
            &mut copies,
            RelocationReason::MemoryPressure,
            &mut ErrorGraph::default(),
        );
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 11 * MIB,
                pages: 1,
                outdated: 0
            },
            "the outdated pool was emptied into the room the current page has"
        );
        assert_eq!(page_of(&kept), page_of(&large));
        assert_eq!(relocation(&memory), None, "nothing is outdated any more");
    }

    /// The bytes the pools hold in pages, every pool together.
    fn reserved(memory: &MemoryManagement<BytesStorage>) -> u64 {
        memory
            .memory_report()
            .dynamic
            .iter()
            .map(|pool| pool.usage.bytes_reserved)
            .sum()
    }

    /// A buffer kept outside a capture, which the next graph reads — a recording's input — takes
    /// no arena page of its own once the stream's graphs pin every page the arena holds. Each such
    /// page would hold that one buffer, and the graph reading it would pin it in turn, so every
    /// recording would add a page of the arena's size, however small its input.
    #[test]
    fn a_kept_input_takes_no_arena_page_once_graphs_pin_the_arena() {
        let mut memory = adaptive();
        // A graph's temporary sizes the arena's pages, and the graph pins its page.
        let temporary = reserve(&mut memory, 8 * MIB);
        let _graph = memory.guard(temporary.descriptor().location());
        drop(temporary);
        let before = reserved(&memory);

        // Each later graph's input, read by that graph and kept for its life.
        let inputs: alloc::vec::Vec<_> = (0..4)
            .map(|_| {
                let input = reserve(&mut memory, 128 * 1024);
                let graph = memory.guard(input.descriptor().location());
                (input, graph)
            })
            .collect();

        let grown = reserved(&memory) - before;
        assert!(
            grown < 9 * MIB,
            "four 128 KiB inputs took {grown} B, an arena page each"
        );
        drop(inputs);
    }

    /// Metadata a graph keeps and small inputs it reads, kept for its life, pack together rather
    /// than taking a metadata page each: once graphs pin every page of the small-allocation pool,
    /// each would otherwise take a page of its own, which the graph reading it pins in turn and
    /// which never empties, a page per graph.
    #[test]
    fn kept_metadata_and_small_inputs_take_no_small_page_per_graph() {
        let mut memory = adaptive();
        let failures = &mut ErrorGraph::default();
        // A graph's temporary metadata, on the small pool's one page, which the graph pins.
        let temporary = reserve_kind(
            &mut memory,
            KIB,
            PageUpdate::Allow,
            AllocationKind::GraphIntermediate,
        )
        .unwrap();
        let _graph = memory.guard(temporary.descriptor().location());
        drop(temporary);
        let mut kept = alloc::vec::Vec::new();
        let before = reserved(&memory);

        for _ in 0..4 {
            // What a recording keeps for its graph: its launches' metadata and its input.
            let metadata = memory
                .reserve_as(
                    KIB,
                    PageUpdate::Allow,
                    AllocationKind::GraphOutput,
                    failures,
                )
                .unwrap();
            let input = reserve(&mut memory, 4 * KIB);
            for handle in [metadata, input] {
                let graph = memory.guard(handle.descriptor().location());
                kept.push((handle, graph));
            }
        }

        let grown = reserved(&memory) - before;
        assert!(
            grown <= 8 * MIB,
            "four graphs' metadata and small inputs took {grown} B, a metadata page each"
        );
    }

    /// Reserve `size` bytes of `kind`, changing the pages held as far as `update` allows.
    fn reserve_kind(
        memory: &mut MemoryManagement<BytesStorage>,
        size: u64,
        update: PageUpdate,
        kind: AllocationKind,
    ) -> Result<ManagedMemoryHandle, IoError> {
        memory.reserve_as(size, update, kind, &mut ErrorGraph::default())
    }

    /// Work outside a capture takes the room graphs' outputs keep apart only once graphs pin the
    /// arena: while the arena has a page nothing pins, a reservation that may not add one finds
    /// no room, where one that may adds an arena page.
    #[test]
    fn eager_work_takes_no_kept_apart_room_while_the_arena_is_unpinned() {
        let mut memory = adaptive();
        // A 9 MiB arena page nothing pins, with 1 MiB left, and an output's 2 MiB page apart.
        let _held = reserve(&mut memory, 8 * MIB);
        let _output = reserve_kind(
            &mut memory,
            128 * KIB,
            PageUpdate::Allow,
            AllocationKind::GraphOutput,
        );

        let placed = reserve_kind(
            &mut memory,
            3 * MIB / 2,
            PageUpdate::Forbidden,
            AllocationKind::Eager,
        );

        assert!(
            matches!(placed, Err(IoError::PageUpdateForbidden { .. })),
            "placed on {:?}",
            placed.as_ref().map(place)
        );
    }

    /// Small work outside a capture takes the room kept apart for graphs' metadata only once
    /// graphs pin every page of the small pool: a full page nothing pins gets a new small page.
    #[test]
    fn small_eager_work_takes_no_kept_apart_room_while_the_small_pool_is_unpinned() {
        let mut memory = adaptive();
        let _full: alloc::vec::Vec<_> = (0..128).map(|_| reserve(&mut memory, 64 * KIB)).collect();
        let metadata = reserve_kind(
            &mut memory,
            KIB,
            PageUpdate::Allow,
            AllocationKind::GraphOutput,
        )
        .unwrap();

        let input = reserve(&mut memory, 4 * KIB);

        assert_ne!(
            input.descriptor().location().pool,
            metadata.descriptor().location().pool
        );
    }

    /// Work outside a capture never grows the separate arena's pages, whose slots nothing frees:
    /// once graphs pin the arena, a buffer the separate arena's pages do not fit takes an arena
    /// page, as it would before.
    #[test]
    fn eager_work_never_grows_the_separate_arena() {
        let mut memory = adaptive();
        // A graph's temporary sizes the arena's pages at 9 MiB, and the graph pins its page. The
        // separate arena carves at its 2 MiB floor.
        let temporary = reserve(&mut memory, 8 * MIB);
        let _graph = memory.guard(temporary.descriptor().location());
        drop(temporary);

        let input = reserve(&mut memory, 3 * MIB);

        assert!(
            (super::ARENA_POOLS..super::SEPARATE_POOLS)
                .contains(&input.descriptor().location().pool),
            "placed on {:?}",
            place(&input)
        );
    }

    /// A small buffer a planning run carved on the room a graph pinned, and that the work kept,
    /// moves to the small pool kept apart with its bytes, as an arena buffer moves to the
    /// separate arena: left there, the graph's next replay would write over it.
    #[test]
    fn a_kept_small_buffer_on_a_pinned_page_moves_apart() {
        let mut memory = adaptive();
        let temporary = reserve_kind(
            &mut memory,
            KIB,
            PageUpdate::Allow,
            AllocationKind::GraphIntermediate,
        )
        .unwrap();
        let _graph = memory.guard(temporary.descriptor().location());
        drop(temporary);
        let kept = reserve_kind(
            &mut memory,
            4 * KIB,
            PageUpdate::Allow,
            AllocationKind::GraphIntermediate,
        )
        .unwrap();
        fill(&mut memory, &kept, 7);

        memory
            .move_apart(
                &[kept.clone().binding()],
                &mut HostCopies,
                &mut ErrorGraph::default(),
            )
            .unwrap();

        assert_eq!(
            (
                kept.descriptor().location().pool,
                contents(&mut memory, &kept)
            ),
            (super::SMALL_SEPARATE_POOL, 7)
        );
    }

    /// With nothing outdated there is nothing to free, however full the
    /// device: no relocation, and no device wait to pay for one.
    #[test]
    fn a_full_device_with_nothing_outdated_asks_for_nothing() {
        let mut memory = adaptive_on_device(4 * MIB);
        let _held = reserve(&mut memory, MIB);
        assert_eq!(relocation(&memory), None);
    }

    /// A relocation that found nothing to move is not planned again until
    /// the arena changes: the next reservations under the same pressure pay
    /// for no plan.
    #[test]
    fn a_relocation_that_moved_nothing_is_not_planned_again() {
        let mut memory = adaptive_on_device(24 * MIB);

        let recorded = reserve(&mut memory, MIB);
        let guard = memory.guard(recorded.descriptor().location());
        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocation(&memory), Some(RelocationReason::MemoryPressure));

        memory.relocate(
            &mut HostCopies,
            RelocationReason::MemoryPressure,
            &mut ErrorGraph::default(),
        );
        assert_eq!(relocation(&memory), None, "the guarded page could not move");

        drop(guard);
        assert_eq!(
            relocation(&memory),
            Some(RelocationReason::MemoryPressure),
            "a released guard is a page the last plan could not move"
        );
    }

    /// A new page is room the last plan did not have, so it plans again.
    #[test]
    fn a_new_page_lets_a_stalled_relocation_plan_again() {
        let mut memory = adaptive_on_device(24 * MIB);

        let recorded = reserve(&mut memory, MIB);
        let _guard = memory.guard(recorded.descriptor().location());
        let _large = reserve(&mut memory, 10 * MIB);
        memory.relocate(
            &mut HostCopies,
            RelocationReason::MemoryPressure,
            &mut ErrorGraph::default(),
        );
        assert_eq!(relocation(&memory), None);

        let _more = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocation(&memory), Some(RelocationReason::MemoryPressure));
    }

    /// An add-only reservation releases nothing, even an outdated page that
    /// emptied: while a graph records, the pages it touched must keep their
    /// numbers and addresses until it seals.
    #[test]
    fn an_add_only_reservation_releases_nothing() {
        let mut memory = adaptive();

        let first = reserve(&mut memory, MIB);
        let _large = reserve(&mut memory, 10 * MIB);
        drop(first);

        let _kept = memory
            .reserve(MIB, PageUpdate::AddOnly, &mut ErrorGraph::default())
            .unwrap();
        assert_eq!(pool(&memory).outdated, 1, "the emptied page is still held");

        let _tick = reserve(&mut memory, MIB);
        assert_eq!(pool(&memory).outdated, 0, "a plain reservation releases it");
    }

    /// The page size follows the largest allocation the pool served: a
    /// megabyte of slack, rounded to the megabyte, never below the floor.
    #[test]
    fn the_page_size_follows_the_largest_allocation() {
        let mut memory = adaptive();

        let _small = reserve(&mut memory, MIB);
        assert_eq!(
            pool(&memory).page_size,
            FLOOR,
            "the floor holds small workloads"
        );

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(pool(&memory).page_size, 11 * MIB);

        let _smaller = reserve(&mut memory, 6 * MIB);
        assert_eq!(
            pool(&memory).page_size,
            11 * MIB,
            "the page size never shrinks"
        );
    }

    /// The page size stops at the largest page the device allocates, and an
    /// allocation no such page fits is no pool's to serve.
    #[test]
    fn the_page_size_stops_at_the_devices_largest_page() {
        const MAX_PAGE: u64 = 16 * MIB;
        let mut memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties::new(MAX_PAGE, 32),
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            MemoryManagementOptions::new("adaptive"),
        );

        let _largest = reserve(&mut memory, MAX_PAGE);
        assert_eq!(
            pool(&memory).page_size,
            MAX_PAGE,
            "the slack does not push the page past the device's limit"
        );

        let refused = memory.reserve(MAX_PAGE + 1, PageUpdate::Allow, &mut ErrorGraph::default());
        assert!(
            matches!(refused, Err(IoError::BufferTooBig { .. })),
            "{refused:?}"
        );
    }

    /// An allocation no page fits is refused before anything changes: the
    /// pages held keep serving. (A refusal by the device itself is the
    /// arena's to test, with a storage that refuses.)
    #[test]
    fn an_allocation_no_page_fits_leaves_the_pages_alone() {
        let mut memory = adaptive();
        let _small = reserve(&mut memory, MIB);

        let refused = memory.reserve(1 << 50, PageUpdate::Allow, &mut ErrorGraph::default());
        assert!(
            matches!(refused, Err(IoError::BufferTooBig { .. })),
            "{refused:?}"
        );
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: FLOOR,
                pages: 1,
                outdated: 0
            }
        );

        let _served = reserve(&mut memory, MIB);
        assert_eq!(pool(&memory).pages, 1, "served from the page already held");
    }

    /// Growing outdates every page held: none of them serves another
    /// reservation, even with room to spare.
    #[test]
    fn a_grown_page_size_serves_nothing_from_outdated_pages() {
        let mut memory = adaptive();

        let first = reserve(&mut memory, MIB);
        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 11 * MIB,
                pages: 2,
                outdated: 1
            }
        );

        let small = reserve(&mut memory, MIB);
        assert_ne!(
            page_of(&small),
            page_of(&first),
            "the outdated page has room to spare, but it is outdated"
        );
    }

    /// An outdated page goes back to the driver on the tick after its last
    /// slice is freed.
    #[test]
    fn an_outdated_page_is_released_once_empty() {
        let mut memory = adaptive();

        let first = reserve(&mut memory, MIB);
        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 11 * MIB,
                pages: 2,
                outdated: 1
            }
        );

        drop(first);
        let _tick = reserve(&mut memory, MIB);
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 11 * MIB,
                pages: 1,
                outdated: 0
            }
        );
    }

    /// A reservation that may not update the pages serves only from the room
    /// they hold, and says so when that room is not enough.
    #[test]
    fn a_forbidden_page_update_serves_only_the_room_held() {
        let mut memory = adaptive();
        let first = reserve(&mut memory, MIB);
        drop(first);

        let _held = memory
            .reserve(MIB, PageUpdate::Forbidden, &mut ErrorGraph::default())
            .expect("the page held has room for it");
        let refused = memory.reserve(4 * MIB, PageUpdate::Forbidden, &mut ErrorGraph::default());
        assert!(
            matches!(refused, Err(IoError::PageUpdateForbidden { .. })),
            "{refused:?}"
        );
        assert_eq!(pool(&memory).pages, 1, "no page was added");
    }

    /// Relocation moves live allocations off outdated pages with their bytes,
    /// their owners resolve to the new place, and the outdated pages go back.
    #[test]
    fn relocation_moves_live_allocations_to_current_pages() {
        let mut memory = adaptive();

        let kept = reserve(&mut memory, MIB);
        fill(&mut memory, &kept, 7);
        let large = reserve(&mut memory, 10 * MIB);
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 11 * MIB,
                pages: 2,
                outdated: 1
            }
        );

        assert_eq!(relocate(&mut memory), 1);

        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 11 * MIB,
                pages: 1,
                outdated: 0
            },
            "the outdated page is gone"
        );
        assert_eq!(contents(&mut memory, &kept), 7, "the bytes moved with it");
        assert_eq!(
            page_of(&kept),
            page_of(&large),
            "it shares the current page"
        );
        assert_eq!(memory.memory_report().usage().bytes_in_use, 11 * MIB);
    }

    /// A plan dropped before its commit leaves every allocation where it was.
    #[test]
    fn an_abandoned_relocation_loses_nothing() {
        let mut memory = adaptive();

        let kept = reserve(&mut memory, MIB);
        fill(&mut memory, &kept, 3);
        let location = place(&kept);
        let _large = reserve(&mut memory, 10 * MIB);

        let mut refused = RefusedCopies;
        memory.relocate(
            &mut refused,
            RelocationReason::Explicit,
            &mut ErrorGraph::default(),
        );

        assert_eq!(place(&kept), location);
        assert_eq!(contents(&mut memory, &kept), 3);
    }

    /// A guarded page keeps its address: a recorded graph replays against it.
    #[test]
    fn relocation_leaves_guarded_pages_in_place() {
        let mut memory = adaptive();

        let recorded = reserve(&mut memory, MIB);
        let location = place(&recorded);
        let _guard = memory.guard(recorded.descriptor().location());

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocate(&mut memory), 0);
        assert_eq!(place(&recorded), location);
    }

    /// A guard holds the whole page: everything on it stays.
    #[test]
    fn a_guard_holds_its_whole_page() {
        let mut memory = adaptive();

        let recorded = reserve(&mut memory, MIB);
        let neighbour = reserve(&mut memory, MIB / 2);
        let location = place(&neighbour);
        assert_eq!(page_of(&neighbour), page_of(&recorded));
        let _guard = memory.guard(recorded.descriptor().location());

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocate(&mut memory), 0);
        assert_eq!(place(&neighbour), location);
    }

    /// A guarded page hands out nothing new, even where its memory is free.
    #[test]
    fn a_guarded_page_serves_no_new_reservation() {
        let mut memory = adaptive();

        let recorded = reserve(&mut memory, MIB);
        let guard = memory.guard(recorded.descriptor().location());
        drop(recorded);

        let later = reserve(&mut memory, MIB);
        assert_eq!(
            pool(&memory).pages,
            2,
            "the guarded page stays out of reach"
        );
        drop(guard);
        drop(later);
    }

    /// Only what the pool serves counts: persistent and dedicated allocations,
    /// however large, leave its page size alone.
    #[test]
    fn other_pools_do_not_move_the_statistic() {
        let mut memory = adaptive();
        let _dynamic = reserve(&mut memory, MIB);

        memory.mode(MemoryAllocationMode::Persistent);
        let _weight = reserve(&mut memory, 100 * MIB);
        memory.mode(MemoryAllocationMode::Auto);

        memory.mode(MemoryAllocationMode::Dedicated);
        let _probe = reserve(&mut memory, 200 * MIB);
        memory.mode(MemoryAllocationMode::Auto);

        assert_eq!(
            pool(&memory),
            Pool {
                page_size: FLOOR,
                pages: 1,
                outdated: 0
            }
        );
    }

    /// A dedicated allocation is its own device allocation, returned on the
    /// tick after it is freed, whatever mode encloses it.
    #[test]
    fn dedicated_allocations_are_released_once_freed() {
        let mut memory = adaptive();

        memory.mode(MemoryAllocationMode::Persistent);
        memory.mode(MemoryAllocationMode::Dedicated);
        let probe = reserve(&mut memory, 200 * MIB);
        memory.mode(MemoryAllocationMode::Auto);
        let weight = reserve(&mut memory, MIB);
        memory.mode(MemoryAllocationMode::Auto);

        assert_eq!(
            memory.memory_report().persistent.usage.bytes_in_use,
            MIB,
            "closing the dedicated window restores the persistent one"
        );
        let report = memory.memory_report();
        assert_eq!(report.usage().bytes_reserved, 200 * MIB + MIB);
        assert_eq!(
            (report.dedicated.kind, report.dedicated.pages),
            (MemoryPoolKind::Direct, 1),
            "the dedicated allocation is reported as its own pool"
        );

        drop(probe);
        let _tick = reserve(&mut memory, MIB);
        assert_eq!(
            memory.memory_report().usage().bytes_reserved,
            MIB + FLOOR,
            "the probe buffer is gone; the weight and one adaptive page remain"
        );
        drop(weight);
    }

    /// A guard's claim ends with the guard: the page can move again.
    #[test]
    fn a_dropped_guard_lets_the_page_move() {
        let mut memory = adaptive();

        let recorded = reserve(&mut memory, MIB);
        drop(memory.guard(recorded.descriptor().location()));

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocate(&mut memory), 1, "the page moves once unguarded");
        drop(recorded);
    }
}
