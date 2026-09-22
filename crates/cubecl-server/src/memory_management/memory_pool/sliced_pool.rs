use crate::{
    memory_management::{
        BytesFormat, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation,
        MemoryPoolKind, MemoryPoolReport, MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool, PageMapping, Slice},
    },
    server::IoError,
    storage::{ComputeStorage, StorageId},
};
use alloc::vec::Vec;
use core::fmt::Display;
use cubecl_environment::backtrace::BackTrace;

/// A pool that carves slices out of pages of one size.
///
/// The size is the pool's for its life: what a workload's allocations grow
/// into is the adaptive memory's to answer, with a pool per size.
pub struct SlicedPool {
    pages: Vec<(MemoryPage, StorageId)>,
    pages_tmp: Vec<(MemoryPage, StorageId)>,
    /// The size every page is allocated at.
    page_size: u64,
    /// The largest allocation the pool accepts.
    max_slice_size: u64,
    /// Whether an allocation past `max_slice_size` but close to the page size
    /// is accepted too, as one that leaves little of its page unused.
    near_page_size: bool,
    alignment: u64,
    location_base: MemoryLocation,
    /// The most pages ever held at once.
    pages_peak: u64,
    /// The largest allocation served, in requested (pre-padding) bytes.
    largest_alloc: u64,
}

impl SlicedPool {
    /// A pool of `page_size` pages.
    pub fn new(page_size: u64, max_slice_size: u64, alignment: u64, pool_pos: u8) -> Self {
        Self {
            pages: Vec::new(),
            pages_tmp: Vec::new(),
            page_size,
            max_slice_size: max_slice_size.min(page_size),
            near_page_size: true,
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            pages_peak: 0,
            largest_alloc: 0,
        }
    }

    /// Accept only up to `max_slice_size`, never an allocation for being close
    /// to the page size: for a pool routed ahead of one whose pages are sized
    /// to what they serve, where those allocations fragment nothing.
    pub fn up_to_max_slice(mut self) -> Self {
        self.near_page_size = false;
        self
    }

    /// The size the pool allocates its pages at.
    pub(crate) fn page_size(&self) -> u64 {
        self.page_size
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self, kind: MemoryPoolKind) -> MemoryPoolReport {
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

    /// Its shape, as a report states it.
    pub(crate) fn kind(&self) -> MemoryPoolKind {
        MemoryPoolKind::Sliced {
            page_size: self.page_size,
            max_slice_size: self.max_slice_size,
        }
    }

    /// How many pages it holds.
    pub(crate) fn pages_held(&self) -> u64 {
        self.pages.len() as u64
    }

    /// The pages it holds.
    #[cfg(multi_threading)]
    pub(crate) fn pages(&self) -> impl Iterator<Item = &MemoryPage> {
        self.pages.iter().map(|(page, _)| page)
    }

    /// The slice `location` names, which the pool is holding.
    #[cfg(multi_threading)]
    pub(crate) fn slice_at(&mut self, location: MemoryLocation) -> &mut Slice {
        self.pages[location.page as usize]
            .0
            .slice_mut(location.slice as usize)
    }

    /// Whether the page `location` names has real device backing.
    #[cfg(multi_threading)]
    pub(crate) fn is_mapped_at(&self, location: MemoryLocation) -> bool {
        self.pages[location.page as usize].0.is_mapped()
    }

    /// Give the page `location` names real device backing, if it was carved
    /// lazily.
    #[cfg(multi_threading)]
    pub(crate) fn map_page_at<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        location: MemoryLocation,
    ) -> Result<(), IoError> {
        self.map_page(storage, location.page as usize)
    }

    /// Reserve `size` bytes on a page already held, coalescing as it goes.
    fn reserve_free(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        let handle = self.pages.iter_mut().find_map(|(page, _)| {
            page.coalesce(failures);
            page.try_reserve(size)
        });
        if handle.is_some() {
            self.largest_alloc = self.largest_alloc.max(size);
        }
        handle
    }

    /// Allocate a page and reserve `size` bytes on it.
    fn alloc_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let mut location_base = self.location_base;
        location_base.page = self.pages.len() as u16;

        // A lazy page gets a minted id with no device memory behind it: it
        // carves, coalesces and counts toward the high-water exactly like a
        // real one, and is rebound to a real allocation on first resolution
        // (`materialize`).
        let handle = mapping.storage_handle(storage, self.page_size)?;
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

    /// Return every page nothing is live on to the driver, and renumber the
    /// rest.
    pub(crate) fn release_empty<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        for (mut page, id) in self.pages.drain(..) {
            page.coalesce(failures);
            if page.is_empty() {
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

    /// Whether it holds no page.
    pub(crate) fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

impl MemoryPool for SlicedPool {
    fn accept(&self, size: u64) -> bool {
        self.max_slice_size >= size
            ||
            // If the size is close to the page size so it doesn't create much fragmentation with
            // unused space.
            (self.near_page_size
                && match self.page_size.checked_sub(size) {
                    Some(diff) => diff * 5 < self.page_size, // 20 % unused space is the max allowed.
                    None => false,
                })
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
        self.reserve_free(size, failures)
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
        let _ = failures;
        self.alloc_page(storage, size, mapping)
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
        if explicit {
            self.release_empty(storage, failures);
        }
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

        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.max_slice_size)
        ))?;
        f.write_str("\n")?;

        for (page, id) in self.pages.iter() {
            let summary = page.summary(false);
            f.write_fmt(format_args!(
                "   - Page {id} num_slices={} => {} free - {} full - {} total\n",
                summary.num_total,
                BytesFormat::new(summary.amount_free),
                BytesFormat::new(summary.amount_full),
                BytesFormat::new(summary.amount_total),
            ))?;
        }

        f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))
    }
}
