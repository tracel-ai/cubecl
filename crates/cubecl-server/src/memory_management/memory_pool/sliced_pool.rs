#[cfg(multi_threading)]
use crate::memory_management::relocation::{MovablePage, Move, OutdatedPages, StorageCopy};
use crate::{
    memory_management::{
        BytesFormat, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation,
        MemoryPoolKind, MemoryPoolReport, MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool, PageMapping, Slice, calculate_padding},
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

/// The unit a page size that follows the largest allocation is rounded up to.
const PAGE_GRANULE: u64 = 1024 * 1024;

/// A pool that carves slices out of pages.
///
/// How big a page is is the pool's [`PageSizing`]: fixed for the pool's life,
/// or following the largest allocation the pool has served. A page is
/// *current* when its size is the pool's page size and *outdated* otherwise —
/// which only a page size that grows ever produces. Only current pages serve reservations; an outdated page is
/// returned to the driver once nothing on it is live, or emptied early by
/// [`plan_relocation`](Self::plan_relocation).
pub struct SlicedPool {
    pages: Vec<(MemoryPage, StorageId)>,
    pages_tmp: Vec<(MemoryPage, StorageId)>,
    /// The size new pages are allocated at.
    page_size: u64,
    sizing: PageSizing,
    alignment: u64,
    location_base: MemoryLocation,
    /// The most pages ever held at once.
    pages_peak: u64,
    /// The largest allocation served, in requested (pre-padding) bytes.
    largest_alloc: u64,
}

/// How a [`SlicedPool`] sizes its pages.
enum PageSizing {
    /// One size for the pool's life.
    Fixed {
        /// The largest allocation the pool accepts.
        max_slice_size: u64,
        /// Max number of pages (`floor(max_pool_size / page_size)`); `None`
        /// keeps unbounded growth.
        max_pages: Option<u16>,
        /// Whether an allocation past `max_slice_size` but close to the page
        /// size is accepted too, as one that leaves little of its page unused.
        near_page_size: bool,
    },
    /// `largest + 1 MiB`, MiB-rounded, never below `min_page_size` nor above
    /// `max_page_size`
    /// ([`PoolType::AdaptivePages`](crate::memory_management::PoolType::AdaptivePages)).
    ///
    /// The statistic is the pool's own: only what is routed here moves it, so
    /// neither persistent allocations, the other pools' traffic nor another
    /// stream's pool change the page size. Nothing is kept across runs: a
    /// dry-run warmup brings it up to date without materializing an
    /// allocation.
    FollowsLargest {
        min_page_size: u64,
        /// The largest page the device allocates, alignment-rounded down: an
        /// allocation that does not fit one is not the pool's to serve.
        max_page_size: u64,
    },
}

/// Which pages a release returns to the driver.
#[derive(Clone, Copy)]
enum Release {
    /// Every page nothing is live on.
    Empty,
    /// Every outdated page nothing is live on.
    OutdatedAndEmpty,
}

impl Release {
    fn selects(self, page: &MemoryPage, page_size: u64) -> bool {
        match self {
            Release::Empty => page.is_empty(),
            Release::OutdatedAndEmpty => page.size() != page_size && page.is_empty(),
        }
    }
}

impl SlicedPool {
    /// A pool of fixed-size pages, capped at `max_pool_size` bytes of them.
    pub fn new(
        page_size: u64,
        max_slice_size: u64,
        alignment: u64,
        pool_pos: u8,
        max_pool_size: Option<u64>,
    ) -> Self {
        // A budget smaller than one page shrinks the page to the
        // (alignment-rounded-down) budget, so the cap is honored rather than
        // exceeded by a single page. A budget below the alignment can't fit
        // even the smallest page the device allows, so it yields zero pages:
        // allocations error instead of overshooting the cap.
        let (page_size, max_pages) = match max_pool_size {
            Some(cap) => {
                let page_size = if cap < page_size {
                    (cap / alignment * alignment).max(alignment)
                } else {
                    page_size
                };
                let max_pages = (cap / page_size).min(u16::MAX as u64) as u16;
                (page_size, Some(max_pages))
            }
            None => (page_size, None),
        };

        Self::with_sizing(
            page_size,
            PageSizing::Fixed {
                max_slice_size: max_slice_size.min(page_size),
                max_pages,
                // Not for a capped pool: it is a budget for the allocations
                // `max_slice_size` routes to it, and near-page-size strays would
                // exhaust it (e.g. a small metadata pool whose page size matches
                // an upload staging chunk).
                near_page_size: max_pages.is_none(),
            },
            alignment,
            pool_pos,
        )
    }

    /// Accept only up to `max_slice_size`, never an allocation for being close
    /// to the page size: for a pool routed ahead of one that sizes its pages
    /// to what it serves, where those allocations fragment nothing.
    pub fn up_to_max_slice(mut self) -> Self {
        if let PageSizing::Fixed { near_page_size, .. } = &mut self.sizing {
            *near_page_size = false;
        }
        self
    }

    /// A pool whose page size follows the largest allocation it has served,
    /// starting from `min_page_size` and never past the device's
    /// `max_page_size`.
    pub fn adaptive(min_page_size: u64, max_page_size: u64, alignment: u64, pool_pos: u8) -> Self {
        let max_page_size = (max_page_size / alignment * alignment).max(alignment);
        let min_page_size = min_page_size
            .max(alignment)
            .next_multiple_of(alignment)
            .min(max_page_size);
        Self::with_sizing(
            min_page_size,
            PageSizing::FollowsLargest {
                min_page_size,
                max_page_size,
            },
            alignment,
            pool_pos,
        )
    }

    fn with_sizing(page_size: u64, sizing: PageSizing, alignment: u64, pool_pos: u8) -> Self {
        Self {
            pages: Vec::new(),
            pages_tmp: Vec::new(),
            page_size,
            sizing,
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            pages_peak: 0,
            largest_alloc: 0,
        }
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self) -> MemoryPoolReport {
        let kind = match &self.sizing {
            PageSizing::Fixed {
                max_slice_size,
                max_pages,
                ..
            } => MemoryPoolKind::Sliced {
                page_size: self.page_size,
                max_slice_size: *max_slice_size,
                max_pool_size: max_pages.map(|pages| pages as u64 * self.page_size),
            },
            PageSizing::FollowsLargest { .. } => MemoryPoolKind::Adaptive {
                page_size: self.page_size,
                outdated_pages: self.outdated().count() as u64,
            },
        };
        MemoryPoolReport {
            kind,
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

    /// The page size an allocation of `size` bytes would leave the pool at:
    /// grown when the page size follows the largest allocation and `size` is a
    /// new largest. Nothing changes until a page of that size is allocated, so
    /// an allocation the device refuses leaves the pool as it was.
    fn page_size_after(&self, size: u64) -> u64 {
        match self.sizing {
            PageSizing::FollowsLargest {
                min_page_size,
                max_page_size,
            } if size > self.largest_alloc => page_size_for(size, self.alignment)
                .clamp(min_page_size, max_page_size)
                .max(self.page_size),
            _ => self.page_size,
        }
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
        let handle = self
            .pages
            .iter_mut()
            .filter(|(page, _)| page.size() == page_size)
            .find_map(|(page, _)| {
                page.coalesce(failures);
                page.try_reserve(size)
            });
        if handle.is_some() {
            self.largest_alloc = self.largest_alloc.max(size);
        }
        handle
    }

    /// Allocate a page of `page_size` bytes, reserve `size` bytes on it, and
    /// make `page_size` the pool's — which outdates every page of another size.
    fn alloc_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        page_size: u64,
        mapping: PageMapping,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let mut location_base = self.location_base;
        location_base.page = self.pages.len() as u16;

        // A lazy page gets a minted id with no device memory behind it: it
        // carves, coalesces and counts toward the high-water exactly like a
        // real one, and is rebound to a real allocation on first resolution
        // (`materialize`).
        let handle = mapping.storage_handle(storage, page_size)?;
        self.page_size = page_size;
        let mut page = MemoryPage::new(handle, self.alignment, location_base, mapping);
        let reserved = page
            .try_reserve(size)
            .expect("callers only allocate a page for an allocation that fits it");
        let storage_id = page.storage_id();
        self.pages.push((page, storage_id));
        self.pages_peak = self.pages_peak.max(self.pages.len() as u64);
        self.largest_alloc = self.largest_alloc.max(size);

        Ok(reserved)
    }

    /// Give the page at `index` real device backing, if it was carved lazily.
    ///
    /// The virtual carving *is* the layout: the page is allocated for real
    /// and rebound — every slice keeps its offset, the minted id ceases to
    /// exist (it never reached the driver).
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

    /// Return the pages `release` selects to the driver and renumber the rest,
    /// judging which are outdated against `page_size`.
    fn release<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        release: Release,
        page_size: u64,
        failures: &mut ErrorGraph,
    ) {
        // Only a page size that moves ever outdates a page, and one that stays
        // outdated can hold a long-lived allocation for the rest of the run:
        // rebuilding the page list is only worth it once one is empty, not on
        // every reservation that finds one outdated. Whether a slice is free
        // needs no coalescing, so the check costs no more than a scan.
        if matches!(release, Release::OutdatedAndEmpty)
            && (matches!(self.sizing, PageSizing::Fixed { .. })
                || !self
                    .pages
                    .iter()
                    .any(|(page, _)| release.selects(page, page_size)))
        {
            return;
        }
        for (mut page, id) in self.pages.drain(..) {
            page.coalesce(failures);
            if release.selects(&page, page_size) {
                // A dropped page takes its slices with it, and any failure a
                // free slice still carried is released here rather than
                // leaked. An unmapped page has nothing behind its minted id;
                // handing it to the driver's deferred-free queue would be
                // garbage.
                page.shed(failures);
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

    /// Reserve a slice on a current page for every live allocation on the
    /// outdated pages [the analysis](OutdatedPages::plan) plans to empty, so
    /// those pages can be returned to the driver instead of waiting on their
    /// longest-lived slice. Current pages are left as they are: this empties
    /// outdated pages, it does not pack current ones.
    ///
    /// Nothing moves yet: the caller copies each [`Move`]'s bytes, then hands
    /// them over with [`commit_relocation`](Self::commit_relocation).
    #[cfg(multi_threading)]
    pub(crate) fn plan_relocation<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Vec<Move> {
        let plan = OutdatedPages::new(self.outdated().map(|(page, _)| page)).plan();

        let mut moves = Vec::new();
        for page in plan {
            // A page whose allocations do not all find a target keeps none of
            // them: moving part of what is on it frees nothing. A smaller page
            // later in the plan may still fit what this one did not.
            if let Ok(page_moves) = self.plan_page(storage, mapping, page, failures) {
                moves.extend(page_moves);
            }
        }
        moves
    }

    /// Reserve a target for every allocation on `page`, or for none: the
    /// targets reserved before a refusal are freed with their moves.
    #[cfg(multi_threading)]
    fn plan_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        mapping: PageMapping,
        page: MovablePage,
        failures: &mut ErrorGraph,
    ) -> Result<Vec<Move>, IoError> {
        page.live
            .into_iter()
            .map(|allocation| self.plan_move(storage, mapping, allocation.handle, failures))
            .collect()
    }

    #[cfg(multi_threading)]
    fn plan_move<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        mapping: PageMapping,
        allocation: ManagedMemoryHandle,
        failures: &mut ErrorGraph,
    ) -> Result<Move, IoError> {
        let source = self.locate(&allocation);
        let size = source.storage.size();
        let source_storage = source.storage.clone();
        let source_mapped = self.pages[allocation.descriptor().page()].0.is_mapped();

        let target = match self.reserve_current(size, failures) {
            Some(target) => target,
            None => self.alloc_page(storage, size, self.page_size, mapping)?,
        };
        // The bytes have to land somewhere real.
        if source_mapped {
            self.map_page(storage, target.descriptor().page())?;
        }

        let copy = source_mapped.then(|| StorageCopy {
            source: source_storage,
            target: self.locate(&target).storage.clone(),
        });

        Ok(Move {
            allocation,
            target,
            copy,
        })
    }

    /// Hand a planned allocation over to its target, once its bytes are
    /// there. The source slice is left free on its outdated page, which the
    /// next cleanup returns to the driver.
    #[cfg(multi_threading)]
    pub(crate) fn commit_relocation(&mut self, relocated: Move, failures: &mut ErrorGraph) {
        let Move {
            allocation, target, ..
        } = relocated;
        // Locations are read now, not at planning: releasing pages since may
        // have renumbered them.
        let source = allocation.descriptor().location();
        let destination = target.descriptor().location();
        // Only the slices hold the handles once these go.
        drop(target);
        drop(allocation);

        let [(source_page, _), (target_page, _)] = self
            .pages
            .get_disjoint_mut([source.page as usize, destination.page as usize])
            .expect("a relocation moves an allocation between two held pages");
        source_page
            .slice_mut(source.slice as usize)
            .hand_over(target_page.slice_mut(destination.slice as usize), failures);
    }

    #[cfg(multi_threading)]
    fn locate(&self, handle: &ManagedMemoryHandle) -> &Slice {
        let location = handle.descriptor().location();
        self.pages[location.page as usize]
            .0
            .slice(location.slice as usize)
    }
}

/// The page size an allocation of `size` bytes asks for, when the page size
/// follows the largest allocation: before the pool's floor and ceiling.
fn page_size_for(size: u64, alignment: u64) -> u64 {
    size.saturating_add(PAGE_SLACK)
        .next_multiple_of(PAGE_GRANULE)
        .next_multiple_of(alignment)
}

impl MemoryPool for SlicedPool {
    fn accept(&self, size: u64) -> bool {
        match &self.sizing {
            PageSizing::Fixed {
                max_slice_size,
                near_page_size,
                ..
            } => {
                *max_slice_size >= size
                    ||
                    // If the size is close to the page size so it doesn't create much fragmentation with
                    // unused space.
                    (*near_page_size
                        && match self.page_size.checked_sub(size) {
                            Some(diff) => diff * 5 < self.page_size, // 20 % unused space is the max allowed.
                            None => false,
                        })
            }
            // A page is sized to what it serves, up to the largest page the
            // device allocates.
            PageSizing::FollowsLargest { max_page_size, .. } => {
                size + calculate_padding(size, self.alignment) <= *max_page_size
            }
        }
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
        // A size that grows the page size has no current page to land on: the
        // pages of the size it asks for are yet to be allocated.
        if self.page_size_after(size) != self.page_size {
            return None;
        }
        self.reserve_current(size, failures)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let page_size = self.page_size_after(size);

        // `alloc` is only called after `try_reserve` coalesced every page and
        // found no fit, so hitting the cap here means the working set truly
        // exceeds the budget.
        if let PageSizing::Fixed {
            max_pages: Some(max_pages),
            ..
        } = self.sizing
            && self.pages.len() >= max_pages as usize
        {
            return Err(IoError::PoolCapacityExceeded {
                size,
                capacity: max_pages as u64 * self.page_size,
                in_use: self.get_memory_usage().bytes_in_use,
                backtrace: BackTrace::capture(),
            });
        }

        // Whatever a page the new size outdates no longer holds goes back
        // before the pool grows, so its footprint tracks the working set
        // through a resize. Only empty pages go, so a refused allocation
        // below loses nothing.
        self.release(storage, Release::OutdatedAndEmpty, page_size, failures);
        self.alloc_page(storage, size, page_size, mapping)
    }

    fn materialize<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let page_index = binding.descriptor().page();
        // An out-of-range page is `find`'s error to report, not ours. So is a
        // stale location whose page index a later cleanup reassigned: it names
        // a page this binding has no claim on, and backing that page would
        // allocate device memory for an allocation nobody asked to resolve —
        // the opposite of what a dry run is for.
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

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
        failures: &mut ErrorGraph,
    ) {
        let release = match explicit {
            true => Release::Empty,
            false => Release::OutdatedAndEmpty,
        };
        self.release(storage, release, self.page_size, failures);
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
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

impl Display for SlicedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.pages.is_empty() {
            return Ok(());
        }

        match &self.sizing {
            PageSizing::Fixed {
                max_slice_size,
                max_pages,
                ..
            } => {
                f.write_fmt(format_args!(
                    " - Sliced Pool page_size={} max_alloc_size={}",
                    BytesFormat::new(self.page_size),
                    BytesFormat::new(*max_slice_size)
                ))?;
                if let Some(max_pages) = max_pages {
                    f.write_fmt(format_args!(
                        " max_pool_size={}",
                        BytesFormat::new(*max_pages as u64 * self.page_size)
                    ))?;
                }
            }
            PageSizing::FollowsLargest { .. } => f.write_fmt(format_args!(
                " - Adaptive Pool page_size={} largest_alloc={}",
                BytesFormat::new(self.page_size),
                BytesFormat::new(self.largest_alloc)
            ))?,
        }
        f.write_str("\n")?;

        for (page, id) in self.pages.iter() {
            let summary = page.summary(false);
            f.write_fmt(format_args!(
                "   - Page {id} num_slices={} => {} free - {} full - {} total{}\n",
                summary.num_total,
                BytesFormat::new(summary.amount_free),
                BytesFormat::new(summary.amount_full),
                BytesFormat::new(summary.amount_total),
                if self.is_current(page) {
                    ""
                } else {
                    " (outdated)"
                },
            ))?;
        }

        f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        logging::ServerLogger,
        memory_management::{
            ErrorGraph, ManagedMemoryHandle, MemoryAllocationMode, MemoryConfiguration,
            MemoryManagement, MemoryManagementOptions, MemoryPoolKind, MemoryPoolOptions, PoolType,
            drop_queue::Fence,
            relocation::{CopyQueue, StorageCopy},
        },
        server::{IoError, ServerError},
        storage::{BytesStorage, ComputeStorage},
    };
    use alloc::vec;
    use cubecl_environment::sync::Arc;
    use cubecl_ir::MemoryDeviceProperties;

    const MIB: u64 = 1024 * 1024;
    const PROPERTIES: MemoryDeviceProperties = MemoryDeviceProperties::new(1024 * MIB, 32);

    /// A memory manager whose only pool is adaptive.
    fn adaptive(min_page_size: u64) -> MemoryManagement<BytesStorage> {
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

    fn pool(memory: &MemoryManagement<BytesStorage>) -> Pool {
        let report = &memory.memory_report().dynamic[0];
        let MemoryPoolKind::Adaptive {
            page_size,
            outdated_pages,
        } = report.kind
        else {
            unreachable!("the only pool is adaptive");
        };
        Pool {
            page_size,
            pages: report.pages,
            outdated: outdated_pages,
        }
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

    /// Run a relocation the way a command does, with a host copy standing in
    /// for the device one. Answers how many allocations moved.
    fn relocate(memory: &mut MemoryManagement<BytesStorage>) -> usize {
        let failures = &mut ErrorGraph::default();
        let relocation = memory.relocation(failures);
        let moved = relocation.len();
        let landed = relocation.copy(&mut HostCopies(memory.storage())).unwrap();
        memory.commit_relocation(landed, failures);
        moved
    }

    /// Host copies between the storages of a [`BytesStorage`]: done as soon
    /// as they are enqueued.
    struct HostCopies<'a>(&'a mut BytesStorage);

    impl CopyQueue for HostCopies<'_> {
        type Fence = HostFence;

        fn copy(&mut self, copy: &StorageCopy) -> Result<(), IoError> {
            let source = self.0.get(&copy.source)?;
            let mut target = self.0.get(&copy.target)?;
            target.write().copy_from_slice(source.read());
            Ok(())
        }

        fn fence(&mut self) -> HostFence {
            HostFence
        }
    }

    /// A host copy has landed by the time it returns.
    struct HostFence;

    impl Fence for HostFence {
        fn wait(self) -> Result<(), ServerError> {
            Ok(())
        }
    }

    /// The page size follows the largest allocation the pool served: a
    /// megabyte of slack, rounded to the megabyte, never below the floor.
    #[test]
    fn the_page_size_follows_the_largest_allocation() {
        let mut memory = adaptive(4 * MIB);

        let _small = reserve(&mut memory, MIB);
        assert_eq!(
            pool(&memory).page_size,
            4 * MIB,
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
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::AdaptivePages {
                        min_page_size: 4 * MIB,
                    },
                    dealloc_period: None,
                }],
            },
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
        let mut memory = adaptive(4 * MIB);
        let _small = reserve(&mut memory, MIB);

        // Past the host's address space: the storage refuses it.
        let refused = memory.reserve(1 << 50, &mut ErrorGraph::default());
        assert!(refused.is_err());
        assert_eq!(
            pool(&memory),
            Pool {
                page_size: 4 * MIB,
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
        let mut memory = adaptive(4 * MIB);

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
            "the outdated page has 3 MiB free, but it is outdated"
        );
    }

    /// An outdated page goes back to the driver on the tick after its last
    /// slice is freed.
    #[test]
    fn an_outdated_page_is_released_once_empty() {
        let mut memory = adaptive(4 * MIB);

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
        let mut memory = adaptive(4 * MIB);

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
        let mut memory = adaptive(4 * MIB);

        let kept = reserve(&mut memory, MIB);
        fill(&mut memory, &kept, 3);
        let location = place(&kept);
        let _large = reserve(&mut memory, 10 * MIB);

        let relocation = memory.relocation(&mut ErrorGraph::default());
        assert_eq!(relocation.len(), 1);
        drop(relocation);

        assert_eq!(place(&kept), location);
        assert_eq!(contents(&mut memory, &kept), 3);
    }

    /// An allocation a graph capture resolved keeps its address: the recorded
    /// kernels replay against it.
    #[test]
    fn relocation_leaves_captured_allocations_in_place() {
        let mut memory = adaptive(4 * MIB);

        let recorded = reserve(&mut memory, MIB);
        let location = place(&recorded);
        memory.mark_captured(&recorded.clone().binding());

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocate(&mut memory), 0);
        assert_eq!(place(&recorded), location);
    }

    /// A captured allocation keeps its page, so moving the rest of the page
    /// would copy bytes and free nothing: none of it moves.
    #[test]
    fn a_captured_allocation_holds_its_whole_page() {
        let mut memory = adaptive(4 * MIB);

        let recorded = reserve(&mut memory, MIB);
        memory.mark_captured(&recorded.clone().binding());
        let neighbour = reserve(&mut memory, MIB);
        let location = place(&neighbour);

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocate(&mut memory), 0);
        assert_eq!(place(&neighbour), location);
    }

    /// Only what the pool serves counts: persistent and dedicated allocations,
    /// however large, leave its page size alone.
    #[test]
    fn other_pools_do_not_move_the_statistic() {
        let mut memory = adaptive(4 * MIB);
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
                page_size: 4 * MIB,
                pages: 1,
                outdated: 0
            }
        );
    }

    /// A dedicated allocation is its own device allocation, returned on the
    /// tick after it is freed, whatever mode encloses it.
    #[test]
    fn dedicated_allocations_are_released_once_freed() {
        let mut memory = adaptive(4 * MIB);

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
            MIB + 4 * MIB,
            "the probe buffer is gone; the weight and one adaptive page remain"
        );
        drop(weight);
    }

    /// A graph's claim on an address ends with the allocation: the next
    /// allocation carved in the same slot can move.
    #[test]
    fn a_reused_slot_owes_nothing_to_an_old_capture() {
        let mut memory = adaptive(4 * MIB);

        let recorded = reserve(&mut memory, MIB);
        memory.mark_captured(&recorded.clone().binding());
        drop(recorded);
        let reused = reserve(&mut memory, MIB);

        let _large = reserve(&mut memory, 10 * MIB);
        assert_eq!(relocate(&mut memory), 1, "the reused slot moves");
        drop(reused);
    }
}
