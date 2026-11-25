use super::{
    ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice, SliceAllocator, SliceReserver,
    missing, padded_size, pool_origin, slices_usage,
};
use crate::{
    memory_management::{
        BytesFormat, Cleanup, ErrorGraph, MemoryLocation, MemoryPoolKind, MemoryPoolReport,
        MemoryUsage, PageGuard,
    },
    server::IoError,
    storage::StorageAllocator,
};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;

/// How fast the running average of the sizes served follows the latest one.
const SIZE_AVG_DECAY: f64 = 0.01;

/// How many cleanups find a page free, unused in between, before it goes back.
const DEALLOC_FREE_CHECK_COUNT: u32 = 5;

/// A pool of one allocation per page, pages sized at least to the running
/// average of what it served, so a page is reused by any allocation it fits.
///
/// Only one slice is supported per page, due to the limitations in WGPU where
/// each buffer should only bound with either read only or `read_write` slices
/// but not a mix of both.
pub(crate) struct ExclusiveSliceReserver {
    /// Every page, read by the position a location carries (Position) and
    /// added at the end (Arrival); a reservation scans them all for the free
    /// page that fits and was found free the fewest times.
    pages: Vec<ExclusivePage>,
    alignment: u64,
    /// The reservations an unused page waits through before it goes back.
    dealloc_period: u64,
    /// The reservation count at the last check for unused pages.
    last_dealloc_check: u64,
    max_alloc_size: u64,
    cur_avg_size: f64,
    location_base: MemoryLocation,
    /// The most pages ever held at once (pages come and go with
    /// `dealloc_period`, so the current count understates the peak).
    peak_pages: u64,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc_bytes: u64,
}

/// One page of an [`ExclusiveSliceReserver`], and how long it has been unused.
struct ExclusivePage {
    /// The page's one slice.
    pub slice: Slice,
    /// The bytes the page was allocated at.
    pub alloc_bytes: u64,
    /// The checks that found the page free since it last served.
    pub free_count: u32,
}

/// What an [`ExclusiveSliceReserver`] serves, and the pool index its pages carry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExclusiveLayout {
    /// The largest allocation the pool accepts, a multiple of the alignment.
    pub max_alloc_bytes: u64,
    /// The alignment every page is allocated at.
    pub alignment_bytes: u64,
    /// The reservations an unused page waits through before it is released;
    /// `u64::MAX` keeps it for good.
    pub dealloc_period_count: u64,
    /// The pool index a page's location carries.
    pub pool: u8,
}

impl ExclusiveSliceReserver {
    /// A pool serving what `layout` says.
    ///
    /// # Panics
    ///
    /// When the layout's largest allocation is not a multiple of its alignment.
    pub(crate) fn new(layout: ExclusiveLayout) -> Self {
        assert_eq!(
            layout.max_alloc_bytes % layout.alignment_bytes,
            0,
            "the largest allocation should be a multiple of the alignment, so pages are aligned"
        );

        Self {
            pages: Vec::new(),
            alignment: layout.alignment_bytes,
            dealloc_period: layout.dealloc_period_count,
            last_dealloc_check: 0,
            max_alloc_size: layout.max_alloc_bytes,
            cur_avg_size: layout.max_alloc_bytes as f64 / 2.0,
            location_base: pool_origin(layout.pool),
            peak_pages: 0,
            largest_alloc_bytes: 0,
        }
    }

    /// The free page that has been free the least, among those that fit
    /// `size_bytes`.
    fn free_page_fitting_bytes(&mut self, size_bytes: u64) -> Option<&mut ExclusivePage> {
        self.pages
            .iter_mut()
            .filter(|page| page.alloc_bytes >= size_bytes && page.slice.is_free())
            .min_by_key(|page| page.free_count)
    }

    /// Whether a cleanup at `reservation_count` checks the pages for unused
    /// ones: every one that `cleanup` asks for, and a periodic one once per
    /// fraction of the period, so a page is freed after at most a period.
    fn checks_at(&self, reservation_count: u64, cleanup: Cleanup) -> bool {
        let check_period = self.dealloc_period / u64::from(DEALLOC_FREE_CHECK_COUNT);
        cleanup == Cleanup::Explicit || reservation_count - self.last_dealloc_check >= check_period
    }
}

impl ExclusiveSliceReserver {
    /// Whether `page` stays: a free one is counted as found free once more, and
    /// released once found free often enough, or on an explicit `cleanup`.
    fn keep_used<Storage: StorageAllocator>(
        page: &mut ExclusivePage,
        cleanup: Cleanup,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> bool {
        if !page.slice.is_free() {
            return true;
        }
        page.free_count += 1;
        if page.free_count < DEALLOC_FREE_CHECK_COUNT && cleanup == Cleanup::Periodic {
            return true;
        }
        SliceAllocator.release_whole(storage, &mut page.slice, failures);
        false
    }
}

impl SliceReserver for ExclusiveSliceReserver {
    fn accepts_bytes(&self, size: u64) -> bool {
        self.max_alloc_size >= size
    }

    /// Reserve a free page that fits `size` bytes.
    fn try_reserve(
        &mut self,
        size: u64,
        _failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        self.cur_avg_size =
            self.cur_avg_size * (1.0 - SIZE_AVG_DECAY) + size as f64 * SIZE_AVG_DECAY;
        let padded = padded_size(size, self.alignment)?;
        let page = self.free_page_fitting_bytes(size)?;
        // A page only ever fits the sizes it was allocated for.
        SliceAllocator.fit_whole(&mut page.slice, size, padded);
        page.free_count = page.free_count.saturating_sub(1);
        let handle = page.slice.handle.clone();
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);
        Some(handle)
    }

    /// Always eager: this pool serves the preset layouts and the auxiliary
    /// (staging/uniform/pinned) managers, whose buffers are written right
    /// after reservation — laziness would only move the same allocation one
    /// call later.
    fn alloc<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        _mapping: PageMapping,
        _failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        }
        let alloc_bytes = (self.cur_avg_size as u64)
            .max(size)
            .next_multiple_of(self.alignment);
        let mut slice = Slice::new(
            SliceAllocator.allocate_storage(storage, PageMapping::Eager, alloc_bytes)?,
            0,
        );
        // `size` is at most the largest allocation, a multiple of the
        // alignment, so it pads within range.
        SliceAllocator.fit_whole(&mut slice, size, size.next_multiple_of(self.alignment));
        slice.descriptor().update_location(MemoryLocation {
            page: self.pages.len() as u16,
            ..self.location_base
        });
        let handle = slice.handle.clone();

        self.pages.push(ExclusivePage {
            slice,
            alloc_bytes,
            // Start the allocation at 'almost ready to free'. Every use will decrement this.
            // This means allocations start as "suspected as unused" and over time will be kept for longer.
            free_count: DEALLOC_FREE_CHECK_COUNT - 1,
        });
        self.peak_pages = self.peak_pages.max(self.pages.len() as u64);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);

        Ok(handle)
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        let page = self.pages.get(location.page as usize)?;
        Some(PageGuard::allocation(page.slice.handle.to_binding()))
    }

    fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Exclusive {
                max_alloc_size: self.max_alloc_size,
            },
            usage: MemoryUsage {
                reserved_bytes: self.pages.iter().map(|page| page.alloc_bytes).sum(),
                ..slices_usage(self.pages.iter().map(|page| &page.slice))
            },
            page_count: self.pages.len() as u64,
            peak_page_count: self.peak_pages,
            // This pool only allocates eagerly.
            unmapped_page_count: 0,
            largest_alloc_bytes: self.largest_alloc_bytes,
        }
    }

    /// Count every free page as found free once more, and return those found
    /// free often enough, or every free one on an explicit cleanup.
    fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        reservation_count: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        if !self.checks_at(reservation_count, cleanup) {
            return;
        }
        self.last_dealloc_check = reservation_count;
        self.pages
            .retain_mut(|page| Self::keep_used(page, cleanup, storage, failures));
        for (index, page) in self.pages.iter().enumerate() {
            page.slice.descriptor().update_page(index as u16);
        }
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let page = self
            .pages
            .get_mut(reserved.descriptor().page())
            .expect("a reserved page should be held by the pool that reserved it");
        SliceAllocator.bind(&mut page.slice, assigned, cursor, failures);
        Ok(())
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let index = binding.descriptor().page();
        self.pages
            .get(index)
            .map(|page| &page.slice)
            .ok_or_else(|| missing("page", index))
    }

    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let index = binding.descriptor().page();
        self.pages
            .get_mut(index)
            .map(|page| &mut page.slice)
            .ok_or_else(|| missing("page", index))
    }
}

impl core::fmt::Display for ExclusiveSliceReserver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            " - Exclusive Pool max_alloc_size={}\n",
            BytesFormat::from_byte_count(self.max_alloc_size)
        ))?;

        for page in self.pages.iter() {
            let is_free = page.slice.is_free();
            let size = BytesFormat::from_byte_count(page.slice.effective_size());

            f.write_fmt(format_args!("   - Page {size} is_free={is_free}\n"))?;
        }

        if !self.pages.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.report().usage))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BytesStorage;

    const ALIGNMENT_BYTES: u64 = 32;

    fn pool(dealloc_period_count: u64) -> ExclusiveSliceReserver {
        ExclusiveSliceReserver::new(ExclusiveLayout {
            max_alloc_bytes: 1024,
            alignment_bytes: ALIGNMENT_BYTES,
            dealloc_period_count,
            pool: 0,
        })
    }

    fn alloc(
        pool: &mut ExclusiveSliceReserver,
        storage: &mut BytesStorage,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        pool.alloc(
            storage,
            size,
            PageMapping::Eager,
            &mut ErrorGraph::default(),
        )
    }

    /// A size past the largest allocation is refused rather than allocated.
    #[test_log::test]
    fn an_allocation_past_the_limit_is_refused() {
        let mut storage = BytesStorage::default();
        let mut pool = pool(u64::MAX);

        assert!(!pool.accepts_bytes(1025));
        assert!(matches!(
            alloc(&mut pool, &mut storage, 1025),
            Err(IoError::BufferTooBig { .. })
        ));
    }

    /// A freed page serves any allocation it fits, with that allocation's
    /// padding.
    #[test_log::test]
    fn a_freed_page_serves_any_allocation_it_fits() {
        let mut storage = BytesStorage::default();
        let mut pool = pool(u64::MAX);
        core::mem::drop(alloc(&mut pool, &mut storage, 512).expect("512 bytes should fit"));

        let _reused = pool
            .try_reserve(100, &mut ErrorGraph::default())
            .expect("the freed page should fit 100 bytes");

        let report = pool.report();
        assert_eq!(report.page_count, 1);
        assert_eq!(report.usage.padding_bytes, 28);
    }

    /// A live page serves nothing else.
    #[test_log::test]
    fn a_live_page_serves_nothing_else() {
        let mut storage = BytesStorage::default();
        let mut pool = pool(u64::MAX);
        let _live = alloc(&mut pool, &mut storage, 512).expect("512 bytes should fit");

        assert!(pool.try_reserve(100, &mut ErrorGraph::default()).is_none());
    }

    /// A free page goes back once enough checks found it free, and a live
    /// one stays, renumbered.
    #[test_log::test]
    fn a_page_found_free_often_enough_goes_back() {
        let mut storage = BytesStorage::default();
        let mut pool = pool(u64::from(DEALLOC_FREE_CHECK_COUNT));
        let freed = alloc(&mut pool, &mut storage, 512).expect("512 bytes should fit");
        let live = alloc(&mut pool, &mut storage, 512).expect("512 bytes should fit");
        core::mem::drop(freed);

        pool.cleanup(
            &mut storage,
            1,
            Cleanup::Periodic,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 1);
        assert_eq!(live.descriptor().location().page, 0);
    }

    /// A periodic cleanup before a check is due changes nothing.
    #[test_log::test]
    fn a_cleanup_before_a_check_is_due_keeps_every_page() {
        let mut storage = BytesStorage::default();
        let mut pool = pool(1000);
        core::mem::drop(alloc(&mut pool, &mut storage, 512).expect("512 bytes should fit"));

        pool.cleanup(
            &mut storage,
            1,
            Cleanup::Periodic,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 1);
    }

    /// An explicit cleanup returns every free page at once.
    #[test_log::test]
    fn an_explicit_cleanup_returns_every_free_page() {
        let mut storage = BytesStorage::default();
        let mut pool = pool(u64::MAX);
        core::mem::drop(alloc(&mut pool, &mut storage, 512).expect("512 bytes should fit"));

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Explicit,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 0);
    }

    /// A pool whose largest allocation breaks the alignment is refused.
    #[test_log::test]
    #[should_panic(expected = "a multiple of the alignment")]
    fn a_misaligned_layout_is_refused() {
        ExclusiveSliceReserver::new(ExclusiveLayout {
            max_alloc_bytes: 1000,
            alignment_bytes: ALIGNMENT_BYTES,
            dealloc_period_count: u64::MAX,
            pool: 0,
        });
    }

    /// A guard of a location naming no page is refused.
    #[test_log::test]
    fn a_guard_of_no_page_is_none() {
        let mut pool = pool(u64::MAX);

        assert!(pool.guard(MemoryLocation::new(0, 3, 0)).is_none());
    }
}
