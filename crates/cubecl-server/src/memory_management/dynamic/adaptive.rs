//! Dynamic memory whose pages follow the workload's largest allocation.

use super::{DynamicPool, Pools};
use crate::memory_management::{
    memory_pool::MemoryPool,
    relocation::{CopyQueue, Move, OutdatedPages, PlannerId, Relocation, StorageCopy},
};
use crate::{
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryPoolKind, MemoryPoolOptions, MemoryPoolReport,
        MemoryUsage, PoolType,
        memory_pool::{PageMapping, SlicedPool, calculate_padding},
    },
    server::IoError,
    storage::ComputeStorage,
};
use alloc::{string::String, vec::Vec};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

/// Slack a page keeps past the largest allocation it was sized for, so that
/// allocation still fits once aligned.
const PAGE_SLACK: u64 = 1024 * 1024;

/// The unit a page size is rounded up to.
const PAGE_GRANULE: u64 = 1024 * 1024;

/// Pages sized to the largest allocation served, with a pool per size.
///
/// What a workload allocates decides which pool serves it: allocations too
/// small to slice and the metadata churn have a pool each, and everything else
/// is carved from pages sized to the largest allocation served so far. When an
/// allocation outgrows those pages, the pool holding them is *outdated* — it
/// serves no new reservation, gives each page back as it empties, and is
/// dropped once it holds none — and a pool of the new size takes over.
pub struct AdaptiveMemory {
    pools: Pools,
    /// What the device holds in total, where it says.
    max_memory: Option<u64>,
    /// What a relocation this one planned is stamped with.
    planner: PlannerId,
    /// The pools every workload uses whatever it allocates, first in routing
    /// order: slots `0..fixed`.
    fixed: u8,
    /// How the pages carving the workload's allocations grow.
    growth: Growth,
}

/// How the pool carving a workload's allocations grows with it.
struct Growth {
    /// The smallest page it allocates.
    min_page_size: u64,
    /// The largest page the device allocates, alignment-rounded down: an
    /// allocation no page of that size fits is not the memory's to serve.
    max_page_size: u64,
    alignment: u64,
    /// The pool new allocations land on.
    current: u8,
    /// The pools a growth left behind, draining.
    outdated: Vec<u8>,
}

impl AdaptiveMemory {
    /// The pools `options` asks for, the last of which carves allocations from
    /// pages that grow.
    pub fn new(
        properties: &MemoryDeviceProperties,
        options: Vec<MemoryPoolOptions>,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        let current = options
            .iter()
            .position(|pool| matches!(pool.pool_type, PoolType::AdaptivePages { .. }))
            .expect("an adaptive layout has a pool whose pages grow") as u8;
        let PoolType::AdaptivePages { min_page_size } = options[current as usize].pool_type else {
            unreachable!("the position of an adaptive pool names one");
        };

        Self {
            pools: Pools::new(properties, &options, true, logger, name),
            max_memory: properties.max_memory(),
            planner: PlannerId::new(),
            fixed: current,
            growth: Growth::new(min_page_size, properties, current),
        }
    }

    /// The pool `index` names, while one is there.
    pub fn get(&self, index: usize) -> Option<&DynamicPool> {
        self.pools.get(index)
    }

    /// The pool `index` names, mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DynamicPool> {
        self.pools.get_mut(index)
    }

    /// Reserve `size` bytes on the pool that serves them, growing the pages
    /// that carve allocations when `size` outgrows them.
    ///
    /// # Errors
    ///
    /// As [`Pools::alloc`].
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.grow_for(storage, size, failures);

        if let Some(handle) = self.pools.try_reserve(self.routing(), size, failures) {
            return Ok(handle);
        }
        let routing = self.routing();
        self.pools.alloc(routing, storage, size, mapping, failures)
    }

    /// Reserve `size` bytes in the room the pools already hold, without
    /// growing the pages. `None` when none has room for it.
    pub fn try_reserve(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        self.pools.try_reserve(self.routing(), size, failures)
    }

    /// Whether another page of the size allocations are carved at would leave
    /// the device with less than one to spare.
    ///
    /// What says a relocation is worth its copies *now*: the pages an outdated
    /// pool holds are the room the next one needs, and once the device has
    /// refused a page there is nowhere left to copy an allocation to. Read
    /// against what this memory management holds, so a runtime with a pool per
    /// stream reads its own other streams as headroom it does not have, which
    /// makes the answer early rather than wrong.
    pub fn crowded(&self) -> bool {
        let Some(max_memory) = self.max_memory else {
            return false;
        };
        let page_size = self.current_page_size();
        self.pools.bytes_reserved() + page_size > max_memory.saturating_sub(page_size)
    }

    /// Empty every outdated pool the current pages have room for, and return
    /// the pages that frees.
    ///
    /// A plan whose copies fail is abandoned: every allocation stays where it
    /// was, and the targets it reserved are freed.
    pub fn relocate<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        copier: &mut dyn CopyQueue<Storage>,
        failures: &mut ErrorGraph,
    ) {
        let relocation = Relocation::new(self.planner, self.plan_relocation(storage, failures));
        if relocation.is_empty() {
            return;
        }
        match relocation.copy(storage, copier) {
            Ok(landed) => {
                for relocated in landed.into_moves(self.planner) {
                    self.commit_relocation(relocated, failures);
                }
                self.drain_outdated(storage, failures);
            }
            // Dropping the plan gave every target it reserved back.
            Err(err) => {
                log::warn!("relocating allocations off outdated memory pages abandoned: {err}")
            }
        }
    }

    /// Release what the pools no longer need, and drop every outdated pool
    /// that has drained.
    pub fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
        failures: &mut ErrorGraph,
    ) {
        self.pools.cleanup(storage, alloc_nr, explicit, failures);
        self.drain_outdated(storage, failures);
    }

    /// The usage of every pool held.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.pools.memory_usage()
    }

    /// A report per pool held, in routing order, the outdated ones last. The
    /// pool carving allocations reports how many pages the outdated ones hold.
    pub fn report(&self) -> Vec<MemoryPoolReport> {
        let outdated_pages = self.outdated().map(|pool| pool.pages_held()).sum::<u64>();
        let mut reports = self
            .pools
            .report(self.routing().chain(self.growth.outdated.iter().copied()));
        if let Some(current) = reports.get_mut(self.fixed as usize)
            && let MemoryPoolKind::Sliced { page_size, .. } = current.kind
        {
            current.kind = MemoryPoolKind::Adaptive {
                page_size,
                outdated_pages,
            };
        }
        reports
    }

    /// The pools that serve reservations, in the order they are tried.
    fn routing(&self) -> impl Iterator<Item = u8> + use<> {
        (0..self.fixed).chain(core::iter::once(self.growth.current))
    }

    /// The pools a growth left behind.
    fn outdated(&self) -> impl Iterator<Item = &SlicedPool> {
        self.growth
            .outdated
            .iter()
            .filter_map(|index| match self.pools.get(*index as usize) {
                Some(DynamicPool::Sliced(pool)) => Some(pool),
                _ => None,
            })
    }

    /// The size allocations are carved at.
    fn current_page_size(&self) -> u64 {
        match self.pools.get(self.growth.current as usize) {
            Some(DynamicPool::Sliced(pool)) => pool.page_size(),
            _ => 0,
        }
    }

    /// Put a pool whose pages fit `size` in front of the reservations when the
    /// one carving them no longer does: the pages it holds are outdated from
    /// here on, and drain.
    fn grow_for<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        failures: &mut ErrorGraph,
    ) {
        let outdated = self.growth.current;
        let page_size = match self.pools.sliced_mut(outdated) {
            Some(pool) => match self.growth.page_size_for(size, pool.page_size()) {
                Some(page_size) => {
                    // Whatever the outdated pool no longer holds goes back
                    // before the new one allocates, so the footprint tracks the
                    // working set through a growth.
                    pool.release_empty(storage, failures);
                    page_size
                }
                None => return,
            },
            None => return,
        };

        let growth = &self.growth;
        let slot = self
            .pools
            .insert(|slot| DynamicPool::Sliced(growth.pool(page_size, slot)));
        self.growth.current = slot;
        self.growth.outdated.push(outdated);
    }

    /// Drop every outdated pool that holds no page any more.
    fn drain_outdated<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        let mut drained = Vec::new();
        for index in self.growth.outdated.clone() {
            let Some(pool) = self.pools.sliced_mut(index) else {
                continue;
            };
            // An outdated pool gives a page back the moment it empties, rather
            // than waiting for the explicit cleanup a pool still serving
            // reservations waits for.
            pool.release_empty(storage, failures);
            if pool.is_empty() {
                drained.push(index);
            }
        }
        self.growth
            .outdated
            .retain(|index| !drained.contains(index));
        for index in drained {
            self.pools.remove(index);
        }
    }
}

impl core::fmt::Display for AdaptiveMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.pools)
    }
}

impl Growth {
    fn new(min_page_size: u64, properties: &MemoryDeviceProperties, current: u8) -> Self {
        let alignment = properties.alignment;
        let max_page_size = (properties.max_page_size / alignment * alignment).max(alignment);
        Self {
            min_page_size: min_page_size
                .max(alignment)
                .next_multiple_of(alignment)
                .min(max_page_size),
            max_page_size,
            alignment,
            current,
            outdated: Vec::new(),
        }
    }

    /// A pool carving pages of `page_size`, at `slot`.
    fn pool(&self, page_size: u64, slot: u8) -> SlicedPool {
        // Every size it is routed, so nothing lands here only to be refused for
        // being larger than a slice of the page it fits.
        SlicedPool::new(page_size, page_size, self.alignment, slot)
    }

    /// The page size `size` asks for, when pages of `page_size` no longer fit
    /// it. `None` while they do, and when no page the device allocates fits it.
    fn page_size_for(&self, size: u64, page_size: u64) -> Option<u64> {
        let needed = size + calculate_padding(size, self.alignment);
        if needed <= page_size || needed > self.max_page_size {
            return None;
        }
        Some(
            size.saturating_add(PAGE_SLACK)
                .next_multiple_of(PAGE_GRANULE)
                .next_multiple_of(self.alignment)
                .clamp(self.min_page_size, self.max_page_size),
        )
    }
}

impl AdaptiveMemory {
    /// Reserve a slice on the pool now carving allocations for every live
    /// allocation on the outdated pools' pages that
    /// [the analysis](OutdatedPages::plan) plans to empty, so those pools
    /// drain instead of waiting on their longest-lived slice.
    ///
    /// Only room the pages already held have: a relocation that allocated a
    /// page would spend more than the pages it frees, and it runs where memory
    /// is short. Nothing moves yet — the caller copies each [`Move`]'s bytes,
    /// then hands them over with
    /// [`commit_relocation`](Self::commit_relocation).
    pub fn plan_relocation<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> Vec<Move> {
        let current = self.growth.current;
        let plan = OutdatedPages::new(self.outdated().flat_map(|pool| pool.pages())).plan();

        let mut moves = Vec::new();
        for page in plan {
            // A page whose allocations do not all find a target keeps none of
            // them: moving part of what is on it frees nothing. A page later in
            // the plan may still fit what this one did not.
            let planned: Option<Vec<Move>> = page
                .live
                .into_iter()
                .map(|allocation| self.plan_move(storage, current, allocation.handle, failures))
                .collect();
            moves.extend(planned.unwrap_or_default());
        }
        moves
    }

    /// Reserve a target for `allocation` on the pool carving allocations, and
    /// say what has to be copied into it. `None` when that pool has no room.
    fn plan_move<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        current: u8,
        allocation: ManagedMemoryHandle,
        failures: &mut ErrorGraph,
    ) -> Option<Move> {
        let source = allocation.descriptor().location();
        let pool = self.pools.sliced_mut(source.pool)?;
        let slice = pool.slice_at(source);
        let source_storage = slice.storage.clone();
        let size = slice.storage.size();
        let source_mapped = pool.is_mapped_at(source);

        let pool = self.pools.sliced_mut(current)?;
        let target = pool.try_reserve(size, failures)?;
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
            allocation,
            target,
            copy,
        })
    }

    /// Hand a planned allocation over to its target, once its bytes are there.
    /// The source slice is left free on its outdated pool, which drains it.
    pub fn commit_relocation(&mut self, relocated: Move, failures: &mut ErrorGraph) {
        let Move {
            allocation, target, ..
        } = relocated;
        // Locations are read now, not at planning: a pool that released pages
        // since may have renumbered them.
        let source = allocation.descriptor().location();
        let destination = target.descriptor().location();
        // Only the slices hold the handles once these go.
        drop(target);
        drop(allocation);

        let Some((source_pool, target_pool)) =
            self.pools.sliced_pair(source.pool, destination.pool)
        else {
            return;
        };
        source_pool
            .slice_at(source)
            .hand_over(target_pool.slice_at(destination), failures);
    }
}

// The layout these tests drive is the `Adaptive` preset, which a build
// refusing sub-slicing does not have.
#[cfg(all(test, not(exclusive_memory_only)))]
mod tests {
    use crate::{
        logging::ServerLogger,
        memory_management::{
            ErrorGraph, ManagedMemoryHandle, MemoryAllocationMode, MemoryConfiguration,
            MemoryManagement, MemoryManagementOptions, MemoryPoolKind,
            relocation::{CopyQueue, StorageCopy},
        },
        server::{IoError, ServerError},
        storage::{BytesStorage, ComputeStorage},
    };
    use cubecl_environment::sync::Arc;
    use cubecl_ir::MemoryDeviceProperties;

    const MIB: u64 = 1024 * 1024;
    const PROPERTIES: MemoryDeviceProperties = MemoryDeviceProperties::new(1024 * MIB, 32);

    /// The floor the `Adaptive` preset carves its pages at, until a larger
    /// allocation grows them.
    const FLOOR: u64 = 2 * MIB;

    /// A memory manager laid out by the `Adaptive` preset. Every size these
    /// tests reserve is past the metadata pool's largest slice, so the pool
    /// that grows is the one serving them.
    /// A memory management on a device that holds `max_memory` bytes.
    fn adaptive_on_device(max_memory: u64) -> MemoryManagement<BytesStorage> {
        MemoryManagement::from_configuration(
            BytesStorage::default(),
            &PROPERTIES.with_max_memory(max_memory),
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            MemoryManagementOptions::new("adaptive"),
        )
    }

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
        memory.reserve(size, &mut ErrorGraph::default()).unwrap()
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
        memory.relocate(&mut copies, failures);
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

    /// Host copies between the storages of a [`BytesStorage`]: done as soon as
    /// they are enqueued.
    struct HostCopies;

    impl CopyQueue<BytesStorage> for HostCopies {
        fn wait_device(&mut self) -> Result<(), ServerError> {
            Ok(())
        }

        fn copy(&mut self, storage: &mut BytesStorage, copy: &StorageCopy) -> Result<(), IoError> {
            let source = storage.get(&copy.source)?;
            let mut target = storage.get(&copy.target)?;
            target.write().copy_from_slice(source.read());
            Ok(())
        }

        fn wait_copies(&mut self) -> Result<(), ServerError> {
            Ok(())
        }
    }

    /// A device with room to spare says nothing; one whose next page would
    /// leave it less than another page asks for the outdated pools to be
    /// emptied first, which is the last moment a relocation has somewhere to
    /// copy to.
    #[test]
    fn a_crowded_device_asks_for_a_relocation_before_its_next_page() {
        const MAX_MEMORY: u64 = 24 * MIB;
        let mut memory = adaptive_on_device(MAX_MEMORY);

        let kept = reserve(&mut memory, MIB);
        assert!(!memory.crowded(), "one 2 MiB page of 24 MiB");

        // 13 MiB held over two pools, and the next page is 11 MiB.
        let _large = reserve(&mut memory, 10 * MIB);
        assert!(memory.crowded());

        let mut copies = HostCopies;
        memory.relocate(&mut copies, &mut ErrorGraph::default());
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 11 * MIB,
                pages: 1,
                outdated: 0
            },
            "the outdated pool was emptied into the room the current page has"
        );
        assert_eq!(page_of(&kept), page_of(&_large));
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

        let refused = memory.reserve(MAX_PAGE + 1, &mut ErrorGraph::default());
        assert!(
            matches!(refused, Err(IoError::BufferTooBig { .. })),
            "{refused:?}"
        );
    }

    /// An allocation the device refuses leaves the page size where it was:
    /// grown to the refused size, every page held would be outdated and every
    /// later allocation would need a page of that size too.
    #[test]
    fn a_refused_allocation_does_not_grow_the_page_size() {
        let mut memory = adaptive();
        let _small = reserve(&mut memory, MIB);

        // Past the host's address space: the storage refuses it.
        let refused = memory.reserve(1 << 50, &mut ErrorGraph::default());
        assert!(refused.is_err());
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
        assert_eq!(memory.memory_usage().bytes_in_use, 11 * MIB);
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
        memory.relocate(&mut refused, &mut ErrorGraph::default());

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
        assert_eq!(memory.memory_usage().bytes_reserved, 200 * MIB + MIB);

        drop(probe);
        let _tick = reserve(&mut memory, MIB);
        assert_eq!(
            memory.memory_usage().bytes_reserved,
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
