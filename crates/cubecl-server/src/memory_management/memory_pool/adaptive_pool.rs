use crate::{
    memory_management::{
        BytesFormat, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation,
        MemoryPoolKind, MemoryPoolReport, MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool, PageMapping, Relocation, Slice, StorageCopy},
    },
    server::IoError,
    storage::{ComputeStorage, StorageId},
};
use alloc::vec::Vec;
use core::fmt::Display;
use cubecl_environment::backtrace::BackTrace;

/// Slack a page keeps past the largest allocation it was sized for, so that
/// allocation still fits once aligned.
const PAGE_SLACK: u64 = 1024 * 1024;

/// The unit page sizes are rounded up to.
const PAGE_GRANULE: u64 = 1024 * 1024;

/// The pool's statistic as the active environment records it: the largest
/// allocation the pool has served, so a pool created under an environment
/// that already ran the workload allocates its final pages from the start —
/// no resize, no outdated page, no compaction.
///
/// Keyed by the memory manager's name and the pool's position rather than by
/// device: the largest allocation is decided by the workload's shapes, not by
/// the hardware, and an environment is a record of one workload. Only
/// recorded where there is a file system to keep it; elsewhere every pool
/// measures from nothing.
struct LargestAllocRecord {
    #[cfg(std_io)]
    store: Option<cubecl_environment::persistence::Store<alloc::string::String, u64>>,
    #[cfg(std_io)]
    key: alloc::string::String,
    /// The environment generation the pool's statistic describes.
    generation: u32,
}

impl LargestAllocRecord {
    fn new(#[cfg_attr(not(std_io), allow(unused_variables))] name: &str, pool_pos: u8) -> Self {
        #[cfg(not(std_io))]
        let _ = pool_pos;
        Self {
            #[cfg(std_io)]
            // Kept like the throughput records, the other measurement an
            // environment carries: always, wherever there is a file system.
            store: {
                use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};

                Some(Store::new(
                    StoreOptions::new()
                        .storage(Namespace::scoped("memory", "adaptive-pool"))
                        .cache(CacheOption::Eager),
                ))
            },
            #[cfg(std_io)]
            key: alloc::format!("{name}/{pool_pos}"),
            generation: cubecl_environment::environment::generation(),
        }
    }

    /// What the active environment recorded, if anything.
    fn load(&self) -> Option<u64> {
        #[cfg(std_io)]
        {
            self.store.as_ref()?.get(&self.key).copied()
        }
        #[cfg(not(std_io))]
        {
            None
        }
    }

    /// Record a new largest allocation. Best-effort: a refused write costs the
    /// next pool a resize, never a wrong allocation.
    fn save(&mut self, #[cfg_attr(not(std_io), allow(unused_variables))] largest_alloc: u64) {
        #[cfg(std_io)]
        if let Some(store) = &mut self.store {
            // The store refuses an insert that would change a key's value.
            store.purge_key(&self.key);
            if let Err(err) = store.insert(self.key.clone(), largest_alloc) {
                log::debug!("the adaptive pool's statistic was not recorded: {err:?}");
            }
        }
    }

    /// Whether the environment switched since the statistic was last read —
    /// one relaxed load, cheap enough for every reservation.
    fn switched(&mut self) -> bool {
        let generation = cubecl_environment::environment::generation();
        let switched = generation != self.generation;
        self.generation = generation;
        switched
    }
}

/// A sliced pool whose page size follows the largest allocation it has
/// served ([`PoolType::AdaptivePages`](crate::memory_management::PoolType::AdaptivePages)).
///
/// The statistic is the pool's own: only what is routed here moves it, so
/// neither persistent allocations nor the other pools' traffic change the
/// page size.
///
/// A page is *current* when its size is the pool's page size and *outdated*
/// otherwise. Only current pages serve reservations; an outdated page is
/// returned to the driver once nothing on it is live, or emptied early by
/// [`plan_compaction`](Self::plan_compaction).
pub struct AdaptivePool {
    pages: Vec<(MemoryPage, StorageId)>,
    pages_tmp: Vec<(MemoryPage, StorageId)>,
    /// The size new pages are allocated at, derived from `largest_alloc`.
    page_size: u64,
    min_page_size: u64,
    alignment: u64,
    location_base: MemoryLocation,
    /// The most pages ever held at once.
    pages_peak: u64,
    /// The largest allocation served — or recorded by the active environment,
    /// which the pool starts from — in requested (pre-padding) bytes.
    largest_alloc: u64,
    record: LargestAllocRecord,
}

impl AdaptivePool {
    /// `name` is the memory manager's, which with `pool_pos` keys the pool's
    /// statistic in the active environment.
    pub fn new(min_page_size: u64, alignment: u64, pool_pos: u8, name: &str) -> Self {
        let min_page_size = min_page_size.max(alignment).next_multiple_of(alignment);

        let mut pool = Self {
            pages: Vec::new(),
            pages_tmp: Vec::new(),
            page_size: min_page_size,
            min_page_size,
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            pages_peak: 0,
            largest_alloc: 0,
            record: LargestAllocRecord::new(name, pool_pos),
        };
        pool.adopt_record();
        pool
    }

    /// Take the active environment's statistic as the pool's own: its page
    /// size, or the floor when the environment recorded none. A smaller page
    /// size than the pool runs is honored too — the pages held become outdated
    /// and drain like any others — since a different environment is a
    /// different workload.
    fn adopt_record(&mut self) {
        self.largest_alloc = self.record.load().unwrap_or(0);
        self.page_size = self.page_size_for(self.largest_alloc);
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Adaptive {
                page_size: self.page_size,
                outdated_pages: self.outdated().count() as u64,
            },
            usage: self.get_memory_usage(),
            pages: self.pages.len() as u64,
            pages_peak: self.pages_peak,
            pages_unmapped: self
                .pages
                .iter()
                .filter(|(page, _)| !page.is_mapped())
                .count() as u64,
            largest_alloc: self.largest_alloc,
        }
    }

    /// The page size an allocation of `size` bytes asks for.
    fn page_size_for(&self, size: u64) -> u64 {
        size.saturating_add(PAGE_SLACK)
            .next_multiple_of(PAGE_GRANULE)
            .next_multiple_of(self.alignment)
            .max(self.min_page_size)
    }

    /// Count `size` toward the statistic, growing the page size when it no
    /// longer fits — which outdates every page held.
    fn observe(&mut self, size: u64) {
        if self.record.switched() {
            self.adopt_record();
        }
        if size <= self.largest_alloc {
            return;
        }
        self.largest_alloc = size;
        self.page_size = self.page_size_for(size);
        self.record.save(size);
    }

    fn is_current(&self, page: &MemoryPage) -> bool {
        page.size() == self.page_size
    }

    fn outdated(&self) -> impl Iterator<Item = &(MemoryPage, StorageId)> {
        self.pages
            .iter()
            .filter(|(page, _)| page.size() != self.page_size)
    }

    /// Reserve `size` bytes on a current page, coalescing as it goes.
    fn reserve_current(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        let page_size = self.page_size;
        self.pages
            .iter_mut()
            .filter(|(page, _)| page.size() == page_size)
            .find_map(|(page, _)| {
                page.coalesce(failures);
                page.try_reserve(size)
            })
    }

    /// Allocate a page of the current size and return its index.
    fn alloc_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        mapping: PageMapping,
    ) -> Result<usize, IoError> {
        let mut location_base = self.location_base;
        location_base.page = self.pages.len() as u16;

        let handle = mapping.storage_handle(storage, self.page_size)?;
        let page = MemoryPage::new(handle, self.alignment, location_base, mapping);
        let storage_id = page.storage_id();
        self.pages.push((page, storage_id));
        self.pages_peak = self.pages_peak.max(self.pages.len() as u64);

        Ok(self.pages.len() - 1)
    }

    /// Give the page at `index` real device backing, if it was carved lazily.
    fn map_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        index: usize,
    ) -> Result<(), IoError> {
        let (page, id) = &mut self.pages[index];
        if page.is_mapped() {
            return Ok(());
        }
        let size = page.size();
        let real = storage
            .alloc(size)
            .map_err(|err| IoError::StorageMappingFailed {
                size,
                source: alloc::boxed::Box::new(err),
                backtrace: BackTrace::capture(),
            })?;
        page.rebind_storage(real.id);
        *id = real.id;
        Ok(())
    }

    /// Drop the pages `release` selects, returning them to the driver, and
    /// renumber the rest.
    fn release_pages<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
        release: impl Fn(&MemoryPage, u64) -> bool,
    ) {
        let page_size = self.page_size;
        for (mut page, id) in self.pages.drain(..) {
            if release(&page, page_size) {
                page.shed(failures);
                // An unmapped page has nothing behind its minted id.
                if page.is_mapped() {
                    storage.dealloc(id);
                }
            } else {
                page.update_page(self.pages_tmp.len() as u16);
                self.pages_tmp.push((page, id));
            }
        }
        core::mem::swap(&mut self.pages, &mut self.pages_tmp);
    }

    /// Return every outdated page nothing is live on to the driver.
    fn release_outdated<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        if self.outdated().next().is_none() {
            return;
        }
        self.release_pages(storage, failures, |page, page_size| {
            page.size() != page_size && page.is_empty()
        });
    }

    /// Reserve a slice on a current page for every live allocation still on
    /// an outdated one, so those pages can be returned to the driver instead
    /// of waiting on their longest-lived slice.
    ///
    /// Nothing moves yet: the caller copies each [`Relocation`]'s bytes, then
    /// hands them over with [`commit`](Self::commit). Allocations a captured
    /// graph recorded stay where they are. Stops early when no target can be
    /// allocated; what was planned so far is still valid.
    pub(crate) fn plan_compaction<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Vec<Relocation> {
        let moves: Vec<ManagedMemoryHandle> = self
            .pages
            .iter()
            .filter(|(page, _)| !self.is_current(page))
            .flat_map(|(page, _)| page.movable().map(|index| page.slice(index).handle.clone()))
            .collect();

        let mut relocations = Vec::with_capacity(moves.len());
        for allocation in moves {
            match self.plan_move(storage, mapping, allocation, failures) {
                Ok(relocation) => relocations.push(relocation),
                Err(_) => break,
            }
        }
        relocations
    }

    fn plan_move<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        mapping: PageMapping,
        allocation: ManagedMemoryHandle,
        failures: &mut ErrorGraph,
    ) -> Result<Relocation, IoError> {
        let source = self.locate(&allocation);
        let size = source.storage.size();
        let source_mapped = self.pages[allocation.descriptor().page()].0.is_mapped();
        let source_storage = source.storage.clone();

        let target = match self.reserve_current(size, failures) {
            Some(target) => target,
            None => {
                let index = self.alloc_page(storage, mapping)?;
                self.pages[index]
                    .0
                    .try_reserve(size)
                    .expect("a fresh page of the current size fits any allocation served")
            }
        };
        // The bytes have to land somewhere real.
        if source_mapped {
            self.map_page(storage, target.descriptor().page())?;
        }

        let copy = source_mapped.then(|| StorageCopy {
            source: source_storage,
            target: self.locate(&target).storage.clone(),
        });

        Ok(Relocation {
            allocation,
            target,
            copy,
        })
    }

    /// Hand each planned allocation over to its target, once its bytes are
    /// there. The source slices are left free on their outdated pages, which
    /// the next cleanup returns to the driver.
    pub(crate) fn commit(&mut self, relocations: Vec<Relocation>, failures: &mut ErrorGraph) {
        for Relocation {
            allocation, target, ..
        } in relocations
        {
            // Locations are read now, not at planning: releasing pages since
            // may have renumbered them.
            let source = allocation.descriptor().location();
            let destination = target.descriptor().location();
            debug_assert_ne!(source.page, destination.page);
            // Only the slices hold the handles once these go.
            drop(target);
            drop(allocation);

            let (low, high) = self
                .pages
                .split_at_mut(source.page.max(destination.page) as usize);
            let (source_page, target_page) = match source.page < destination.page {
                true => (&mut low[source.page as usize].0, &mut high[0].0),
                false => (&mut high[0].0, &mut low[destination.page as usize].0),
            };
            source_page
                .slice_mut(source.slice as usize)
                .hand_over(target_page.slice_mut(destination.slice as usize), failures);
        }
    }

    fn locate(&self, handle: &ManagedMemoryHandle) -> &Slice {
        let location = handle.descriptor().location();
        self.pages[location.page as usize]
            .0
            .slice(location.slice as usize)
    }
}

impl MemoryPool for AdaptivePool {
    fn accept(&self, _size: u64) -> bool {
        true
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let page_index = binding.descriptor().page();
        let (page, _) = self
            .pages
            .get(page_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory page {page_index} doesn't exist").into(),
            })?;
        page.find(binding)
    }

    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let page_index = binding.descriptor().page();
        let (page, _) = self
            .pages
            .get_mut(page_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory page {page_index} doesn't exist").into(),
            })?;
        page.find_mut(binding)
    }

    fn try_reserve(&mut self, size: u64, failures: &mut ErrorGraph) -> Option<ManagedMemoryHandle> {
        self.observe(size);
        self.reserve_current(size, failures)
    }

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.observe(size);
        // Whatever an outdated page no longer holds goes back before the pool
        // grows, so its footprint tracks the working set through a resize.
        self.release_outdated(storage, failures);

        let index = self.alloc_page(storage, mapping)?;
        Ok(self.pages[index]
            .0
            .try_reserve(size)
            .expect("a fresh page of the current size fits any allocation served"))
    }

    fn materialize<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let page_index = binding.descriptor().page();
        // An out-of-range page, or a stale location a later cleanup
        // renumbered, is `find`'s error to report — backing it would allocate
        // for an allocation nobody asked to resolve.
        let claimed = self.pages.get(page_index).is_some_and(|(page, _)| {
            page.find(binding)
                .is_ok_and(|slice| slice.handle.descriptor() == binding.descriptor())
        });
        if !claimed {
            return Ok(());
        }
        self.map_page(storage, page_index)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        self.pages
            .iter()
            .fold(MemoryUsage::default(), |usage, (page, _)| {
                usage.combine(page.memory_usage())
            })
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
        failures: &mut ErrorGraph,
    ) {
        match explicit {
            // Every page nothing is live on, current or not.
            true => self.release_pages(storage, failures, |page, _| page.is_empty()),
            false => self.release_outdated(storage, failures),
        }
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let (page, _) = &mut self.pages[reserved.descriptor().page()];
        page.bind(reserved, assigned, cursor, failures)
    }
}

impl Display for AdaptivePool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.pages.is_empty() {
            return Ok(());
        }

        writeln!(
            f,
            " - Adaptive Pool page_size={} largest_alloc={}",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.largest_alloc)
        )?;
        for (page, id) in self.pages.iter() {
            let summary = page.summary(false);
            writeln!(
                f,
                "   - Page {id} ({}) num_slices={} => {} free - {} full{}",
                BytesFormat::new(summary.amount_total),
                summary.num_total,
                BytesFormat::new(summary.amount_free),
                BytesFormat::new(summary.amount_full),
                if self.is_current(page) {
                    ""
                } else {
                    " (outdated)"
                },
            )?;
        }

        write!(f, "\n{}\n", self.get_memory_usage())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        logging::ServerLogger,
        memory_management::{
            ErrorGraph, ManagedMemoryHandle, MemoryAllocationMode, MemoryConfiguration,
            MemoryManagement, MemoryManagementOptions, MemoryPoolKind, MemoryPoolOptions, PoolType,
        },
        storage::{BytesStorage, ComputeStorage},
    };
    use alloc::vec;
    use cubecl_environment::sync::Arc;
    use cubecl_ir::MemoryDeviceProperties;

    const MIB: u64 = 1024 * 1024;
    const PROPERTIES: MemoryDeviceProperties = MemoryDeviceProperties::new(1024 * MIB, 32);

    /// Environments kept under a temporary root for the whole test binary, so
    /// the pools' records never reach a real one — and a run never starts from
    /// what the previous run recorded.
    fn isolated() {
        static ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
        ROOT.get_or_init(|| {
            let root = tempfile::tempdir().unwrap();
            // Loading the runtime config activates the configured environment,
            // so it is loaded first — or it would undo the redirect on first use.
            <crate::config::CubeClRuntimeConfig as crate::config::RuntimeConfig>::get();
            cubecl_environment::environment::set_root(root.path());
            root
        });
    }

    /// A memory manager whose only pool is adaptive. `name` keys the pool's
    /// record, so each test names its own: tests run concurrently in one
    /// environment.
    fn adaptive(name: &str, min_page_size: u64) -> MemoryManagement<BytesStorage> {
        isolated();
        MemoryManagement::from_configuration(
            BytesStorage::default(),
            &PROPERTIES,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::AdaptivePages { min_page_size },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            MemoryManagementOptions::new(name),
        )
    }

    /// The adaptive pool's page size and page counts, as its report states.
    fn pool(memory: &MemoryManagement<BytesStorage>) -> (u64, u64, u64) {
        let report = &memory.memory_report().dynamic[0];
        let MemoryPoolKind::Adaptive {
            page_size,
            outdated_pages,
        } = report.kind
        else {
            unreachable!("the only pool is adaptive");
        };
        (page_size, report.pages, outdated_pages)
    }

    fn reserve(memory: &mut MemoryManagement<BytesStorage>, size: u64) -> ManagedMemoryHandle {
        memory.reserve(size, &mut ErrorGraph::default()).unwrap()
    }

    fn place(handle: &ManagedMemoryHandle) -> (usize, usize) {
        (handle.descriptor().page(), handle.descriptor().slice())
    }

    fn page_of(handle: &ManagedMemoryHandle) -> usize {
        handle.descriptor().page()
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

    /// Run a compaction the way a command does, with a host copy standing in
    /// for the device one.
    fn compact(memory: &mut MemoryManagement<BytesStorage>) -> usize {
        let failures = &mut ErrorGraph::default();
        let relocations = memory.plan_compaction(failures);
        let moved = relocations.len();
        for copy in relocations
            .iter()
            .filter_map(|relocation| relocation.copy.as_ref())
        {
            let source = memory.storage().get(&copy.source).unwrap();
            let mut target = memory.storage().get(&copy.target).unwrap();
            target.write().copy_from_slice(source.read());
        }
        memory.commit_compaction(relocations, failures);
        moved
    }

    /// The page size follows the largest allocation the pool served: a
    /// megabyte of slack, rounded to the megabyte, never below the floor.
    #[test]
    fn the_page_size_follows_the_largest_allocation() {
        let mut memory = adaptive("the_page_size_follows_the_largest_allocation", 4 * MIB);

        let _small = reserve(&mut memory, MIB);
        assert_eq!(pool(&memory).0, 4 * MIB, "the floor holds small workloads");

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(pool(&memory).0, 11 * MIB);

        let _smaller = reserve(&mut memory, 6 * MIB);
        assert_eq!(pool(&memory).0, 11 * MIB, "the page size never shrinks");
    }

    /// Growing outdates every page held: none of them serves another
    /// reservation, even with room to spare.
    #[test]
    fn a_grown_page_size_serves_nothing_from_outdated_pages() {
        let mut memory = adaptive(
            "a_grown_page_size_serves_nothing_from_outdated_pages",
            4 * MIB,
        );

        let first = reserve(&mut memory, MIB);
        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(pool(&memory), (11 * MIB, 2, 1));

        let small = reserve(&mut memory, MIB);
        assert_ne!(
            page_of(&small),
            page_of(&first),
            "the outdated page has 3 MiB free, but it is outdated"
        );
    }

    /// An outdated page goes back to the driver on the tick after its last
    /// slice is freed.
    #[test]
    fn an_outdated_page_is_released_once_empty() {
        let mut memory = adaptive("an_outdated_page_is_released_once_empty", 4 * MIB);

        let first = reserve(&mut memory, MIB);
        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(pool(&memory), (11 * MIB, 2, 1));

        drop(first);
        let _tick = reserve(&mut memory, MIB);
        assert_eq!(pool(&memory), (11 * MIB, 1, 0));
    }

    /// Compaction moves live allocations off outdated pages with their bytes,
    /// their owners resolve to the new place, and the outdated pages go back.
    #[test]
    fn compaction_moves_live_allocations_to_current_pages() {
        let mut memory = adaptive(
            "compaction_moves_live_allocations_to_current_pages",
            4 * MIB,
        );

        let kept = reserve(&mut memory, MIB);
        fill(&mut memory, &kept, 7);
        let large = reserve(&mut memory, 10 * MIB);
        assert_eq!(pool(&memory), (11 * MIB, 2, 1));

        assert_eq!(compact(&mut memory), 1);

        assert_eq!(pool(&memory), (11 * MIB, 1, 0), "the outdated page is gone");
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
    fn an_abandoned_compaction_loses_nothing() {
        let mut memory = adaptive("an_abandoned_compaction_loses_nothing", 4 * MIB);

        let kept = reserve(&mut memory, MIB);
        fill(&mut memory, &kept, 3);
        let location = place(&kept);
        let _large = reserve(&mut memory, 10 * MIB);

        let relocations = memory.plan_compaction(&mut ErrorGraph::default());
        assert_eq!(relocations.len(), 1);
        drop(relocations);

        assert_eq!(place(&kept), location);
        assert_eq!(contents(&mut memory, &kept), 3);
    }

    /// An allocation a graph capture resolved keeps its address: the recorded
    /// kernels replay against it.
    #[test]
    fn compaction_leaves_captured_allocations_in_place() {
        let mut memory = adaptive("compaction_leaves_captured_allocations_in_place", 4 * MIB);

        let recorded = reserve(&mut memory, MIB);
        let location = place(&recorded);
        memory.capture_begin();
        memory.get_storage(recorded.clone().binding()).unwrap();
        let _graph = memory.capture_end();

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(compact(&mut memory), 0);
        assert_eq!(place(&recorded), location);
    }

    /// Only what the pool serves counts: persistent and unpooled allocations,
    /// however large, leave its page size alone.
    #[test]
    fn other_pools_do_not_move_the_statistic() {
        let mut memory = adaptive("other_pools_do_not_move_the_statistic", 4 * MIB);
        let _dynamic = reserve(&mut memory, MIB);

        memory.mode(MemoryAllocationMode::Persistent);
        let _weight = reserve(&mut memory, 100 * MIB);
        memory.mode(MemoryAllocationMode::Auto);

        memory.mode(MemoryAllocationMode::Unpooled);
        let _probe = reserve(&mut memory, 200 * MIB);
        memory.mode(MemoryAllocationMode::Auto);

        assert_eq!(pool(&memory), (4 * MIB, 1, 0));
    }

    /// An unpooled allocation is its own device allocation, returned on the
    /// tick after it is freed, whatever mode encloses it.
    #[test]
    fn unpooled_allocations_are_released_once_freed() {
        let mut memory = adaptive("unpooled_allocations_are_released_once_freed", 4 * MIB);

        memory.mode(MemoryAllocationMode::Persistent);
        memory.mode(MemoryAllocationMode::Unpooled);
        let probe = reserve(&mut memory, 200 * MIB);
        memory.mode(MemoryAllocationMode::Auto);
        let weight = reserve(&mut memory, MIB);
        memory.mode(MemoryAllocationMode::Auto);

        assert_eq!(
            memory.memory_report().persistent.usage.bytes_in_use,
            MIB,
            "closing the unpooled window restores the persistent one"
        );
        assert_eq!(memory.memory_usage().bytes_reserved, 200 * MIB + MIB);

        drop(probe);
        let _tick = reserve(&mut memory, MIB);
        assert_eq!(
            memory.memory_usage().bytes_reserved,
            MIB + 4 * MIB,
            "the probe buffer is gone; the weight and one adaptive page remain"
        );
        drop(weight);
    }

    /// A pool starts from what the active environment recorded, and adopts
    /// another environment's figure — or the floor — when it switches.
    #[test]
    fn the_environment_records_the_page_size() {
        const NAME: &str = "the_environment_records_the_page_size";
        let mut memory = adaptive(NAME, 4 * MIB);
        let _large = reserve(&mut memory, 10 * MIB);
        drop(memory);

        let mut memory = adaptive(NAME, 4 * MIB);
        assert_eq!(
            pool(&memory).0,
            11 * MIB,
            "a new pool starts at the recorded size"
        );
        let small = reserve(&mut memory, MIB);
        assert_eq!(
            pool(&memory),
            (11 * MIB, 1, 0),
            "no resize, no outdated page"
        );
        drop(small);
    }
}
