use crate::{
    memory_management::{
        BytesFormat, Cleanup, ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle,
        MemoryLocation, MemoryPoolKind, MemoryPoolReport, PageGuard,
        memory_pool::{
            PageCarver, PageMapping, Slice, SliceAllocator, SliceReserver, missing, pool_origin,
            sum_usage,
        },
    },
    server::IoError,
    storage::{StorageAllocator, StorageHandle},
};
use alloc::vec::Vec;
use core::fmt::Display;

/// Whether a reservation may carve the free room on a page a graph guards (see [`PageGuard`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnedRoom {
    /// Leave it alone: what lands there would be overwritten by the next replay of the graph
    /// that used that room.
    Avoid,
    /// Carve it too: whatever lands there is overwritten only while it is not in use.
    Share,
}

/// A pool that carves slices out of pages of one size.
///
/// The size is the pool's for its life: what a workload's allocations grow
/// into is the adaptive memory's to answer, with a pool per size.
pub(crate) struct PagedSliceReserver {
    /// Every page, read by the index a location carries (Position) and
    /// scanned in that order for room (Ordered).
    pages: Vec<PageCarver>,
    layout: PagedLayout,
    /// Whether an allocation past the largest slice but close to the page size
    /// is accepted too, as one that leaves little of its page unused.
    near_page_size: bool,
    location_base: MemoryLocation,
    /// The most pages ever held at once.
    peak_pages: u64,
    /// The largest allocation served, in requested (pre-padding) bytes.
    largest_alloc_bytes: u64,
}

/// What a [`PagedSliceReserver`] carves, and the pool index its slices carry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PagedLayout {
    /// The size every page is allocated at.
    pub page_size_bytes: u64,
    /// The largest allocation the pool accepts, capped at the page size.
    pub max_slice_bytes: u64,
    /// The alignment every slice starts at.
    pub alignment_bytes: u64,
    /// The pool index a slice's location carries.
    pub pool: u8,
}

impl PagedSliceReserver {
    /// A pool carving pages as `layout` says.
    pub(crate) fn new(layout: PagedLayout) -> Self {
        Self {
            pages: Vec::new(),
            layout: PagedLayout {
                max_slice_bytes: layout.max_slice_bytes.min(layout.page_size_bytes),
                ..layout
            },
            near_page_size: true,
            location_base: pool_origin(layout.pool),
            peak_pages: 0,
            largest_alloc_bytes: 0,
        }
    }

    /// Accept only up to the largest slice, never an allocation for being
    /// close to the page size: for a pool routed ahead of one whose pages are
    /// sized to what they serve, where those allocations fragment nothing.
    pub(crate) fn up_to_max_slice(mut self) -> Self {
        self.near_page_size = false;
        self
    }

    /// Whether graphs guard every page the pool holds, and it holds some: room
    /// for work outside a capture comes only from a new page then.
    pub(crate) fn is_pinned(&self) -> bool {
        !self.pages.is_empty() && self.pages.iter().all(|page| page.view().is_guarded)
    }

    /// The pages it holds, in the order a location numbers them.
    pub(crate) fn pages(&self) -> &[PageCarver] {
        &self.pages
    }

    /// The page `location` names, which the pool is holding.
    ///
    /// # Panics
    ///
    /// When the pool holds no page there.
    pub(crate) fn page_at(&self, location: MemoryLocation) -> &PageCarver {
        &self.pages[location.page as usize]
    }

    /// The slice `location` names, which the pool is holding.
    ///
    /// # Panics
    ///
    /// When the pool holds no slice there.
    pub(crate) fn slice_at(&mut self, location: MemoryLocation) -> &mut Slice {
        self.pages[location.page as usize].slice_at_index_mut(location.slice as usize)
    }

    /// The storage behind the slice `location` names, giving its page real
    /// device backing first if it was carved lazily.
    ///
    /// # Errors
    ///
    /// [`IoError::StorageMappingFailed`] when the storage refuses the backing.
    ///
    /// # Panics
    ///
    /// When the pool holds no slice there.
    pub(crate) fn mapped_storage_at<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        location: MemoryLocation,
    ) -> Result<StorageHandle, IoError> {
        self.pages[location.page as usize].map(storage)?;
        Ok(self.slice_at(location).storage.clone())
    }

    /// Reserve `size` bytes on a page already held, coalescing as it goes. A
    /// guarded page is carved only when `room` shares it.
    pub(crate) fn try_reserve_sharing(
        &mut self,
        size: u64,
        room: PinnedRoom,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        let handle = self
            .pages
            .iter_mut()
            .filter(|page| room == PinnedRoom::Share || !page.view().is_guarded)
            .find_map(|page| {
                page.coalesce(failures);
                page.carve_bytes(size)
            })?;
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);
        Some(handle)
    }

    /// Reserve `size` bytes as [`try_reserve_sharing`](Self::try_reserve_sharing)
    /// does, allocating a page when the room held has none.
    ///
    /// # Errors
    ///
    /// Whatever the storage refuses.
    ///
    /// # Panics
    ///
    /// When `size` is one [`accept`](SliceReserver::accepts_bytes) refuses.
    pub(crate) fn reserve_sharing<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        room: PinnedRoom,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match self.try_reserve_sharing(size, room, failures) {
            Some(handle) => Ok(handle),
            None => self.alloc_page(storage, size, mapping),
        }
    }

    /// Return every page nothing is live on and nothing guards to the driver,
    /// and renumber the rest.
    pub(crate) fn release_empty<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) {
        self.pages
            .retain_mut(|page| !page.release_if_unused(storage, failures));
        for (index, page) in self.pages.iter_mut().enumerate() {
            page.set_page_index(index as u16);
        }
    }

    /// Allocate a page and reserve `size` bytes on it.
    ///
    /// A lazy page gets a minted id with no device memory behind it: it
    /// carves, coalesces and counts toward the high-water exactly like a real
    /// one, and is rebound to a real allocation on first resolution
    /// (`materialize`).
    fn alloc_page<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let location_base = MemoryLocation {
            page: self.pages.len() as u16,
            ..self.location_base
        };
        let handle =
            SliceAllocator::new(storage).allocate_storage(mapping, self.layout.page_size_bytes)?;
        let mut page = PageCarver::new(handle, self.layout.alignment_bytes, location_base, mapping);
        let reserved = page
            .carve_bytes(size)
            .expect("a page should be allocated only for an allocation the pool accepts");
        self.pages.push(page);
        self.peak_pages = self.peak_pages.max(self.pages.len() as u64);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);

        Ok(reserved)
    }
}

/// Whether `size` bytes leave little enough of a page of `page_size` unused,
/// a fifth at most, to be worth a page of their own.
fn near_page(size: u64, page_size: u64) -> bool {
    page_size
        .checked_sub(size)
        .is_some_and(|unused| unused * 5 < page_size)
}

impl SliceReserver for PagedSliceReserver {
    fn accepts_bytes(&self, size: u64) -> bool {
        self.layout.max_slice_bytes >= size
            || (self.near_page_size && near_page(size, self.layout.page_size_bytes))
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let index = binding.descriptor().page();
        self.pages
            .get(index)
            .ok_or_else(|| missing("page", index))?
            .find(binding)
    }

    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let index = binding.descriptor().page();
        self.pages
            .get_mut(index)
            .ok_or_else(|| missing("page", index))?
            .find_mut(binding)
    }

    fn try_reserve(&mut self, size: u64, failures: &mut ErrorGraph) -> Option<ManagedMemoryHandle> {
        self.try_reserve_sharing(size, PinnedRoom::Avoid, failures)
    }

    fn alloc<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        _failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.alloc_page(storage, size, mapping)
    }

    /// A stale location whose page index a later cleanup reassigned names a
    /// page this binding has no claim on, so backing it would allocate device
    /// memory for an allocation nobody asked to resolve — the opposite of what
    /// discarding launches is for.
    fn materialize<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let Some(page) = self
            .pages
            .get_mut(binding.descriptor().page())
            .filter(|page| {
                page.find(binding)
                    .is_ok_and(|slice| slice.descriptor() == binding.descriptor())
            })
        else {
            return Ok(());
        };
        page.map(storage)
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        Some(self.pages.get_mut(location.page as usize)?.guard())
    }

    fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Sliced {
                page_size: self.layout.page_size_bytes,
                max_slice_size: self.layout.max_slice_bytes,
            },
            usage: sum_usage(self.pages.iter().map(|page| page.view().usage())),
            page_count: self.pages.len() as u64,
            peak_page_count: self.peak_pages,
            unmapped_page_count: self
                .pages
                .iter()
                .filter(|page| !page.view().is_mapped)
                .count() as u64,
            largest_alloc_bytes: self.largest_alloc_bytes,
        }
    }

    /// On an explicit cleanup, return every page nothing is live on and
    /// nothing guards; a periodic one keeps every page for reuse.
    fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        _reservation_count: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        if cleanup == Cleanup::Periodic {
            return;
        }
        self.release_empty(storage, failures);
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
        page.bind(reserved, assigned, cursor, failures);
        Ok(())
    }
}

impl Display for PagedSliceReserver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.pages.is_empty() {
            return Ok(());
        }

        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}",
            BytesFormat::from_byte_count(self.layout.page_size_bytes),
            BytesFormat::from_byte_count(self.layout.max_slice_bytes)
        ))?;
        f.write_str("\n")?;

        for page in self.pages.iter() {
            let summary = page.view().summary_with_blocks(false);
            f.write_fmt(format_args!(
                "   - Page {} num_slices={} => {} free - {} full - {} total\n",
                page.view().storage_id,
                summary.total_count,
                BytesFormat::from_byte_count(summary.free_bytes),
                BytesFormat::from_byte_count(summary.full_bytes),
                BytesFormat::from_byte_count(summary.total_bytes),
            ))?;
        }

        f.write_fmt(format_args!("\n{}\n", self.report().usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BytesStorage;

    const KIB: u64 = 1024;

    fn pool() -> PagedSliceReserver {
        PagedSliceReserver::new(PagedLayout {
            page_size_bytes: 64 * KIB,
            max_slice_bytes: 8 * KIB,
            alignment_bytes: 32,
            pool: 1,
        })
    }

    fn reserve(
        pool: &mut PagedSliceReserver,
        storage: &mut BytesStorage,
        size: u64,
    ) -> ManagedMemoryHandle {
        pool.reserve(
            storage,
            size,
            PageMapping::Eager,
            &mut ErrorGraph::default(),
        )
        .expect("the pool should serve the size")
    }

    /// A size up to the largest slice is accepted.
    #[test_log::test]
    fn a_size_up_to_the_largest_slice_is_accepted() {
        assert!(pool().accepts_bytes(8 * KIB));
    }

    /// A size past the largest slice is accepted only when it leaves little
    /// of a page unused.
    #[test_log::test]
    fn a_size_near_the_page_size_is_accepted() {
        assert!(pool().accepts_bytes(60 * KIB));
        assert!(!pool().accepts_bytes(32 * KIB));
        assert!(!pool().accepts_bytes(65 * KIB));
    }

    /// A pool kept to its largest slice accepts nothing past it.
    #[test_log::test]
    fn a_pool_kept_to_its_largest_slice_refuses_near_page_sizes() {
        assert!(!pool().up_to_max_slice().accepts_bytes(60 * KIB));
    }

    /// The largest slice never passes the page size.
    #[test_log::test]
    fn the_largest_slice_is_capped_at_the_page_size() {
        let pool = PagedSliceReserver::new(PagedLayout {
            page_size_bytes: 64 * KIB,
            max_slice_bytes: 128 * KIB,
            alignment_bytes: 32,
            pool: 1,
        });

        assert!(!pool.accepts_bytes(65 * KIB));
    }

    /// Reservations share a page until it is full, then take a new one.
    #[test_log::test]
    fn reservations_share_a_page_until_it_is_full() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let _held: Vec<_> = (0..9)
            .map(|_| reserve(&mut pool, &mut storage, 8 * KIB))
            .collect();

        assert_eq!(pool.report().page_count, 2);
    }

    /// A guarded page is not carved by a reservation that avoids its room.
    #[test_log::test]
    fn a_guarded_page_is_not_carved_when_avoided() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let first = reserve(&mut pool, &mut storage, KIB);
        let _guard = pool.guard(first.descriptor().location());

        assert!(pool.try_reserve(KIB, &mut ErrorGraph::default()).is_none());
    }

    /// A guarded page is carved by a reservation that shares its room.
    #[test_log::test]
    fn a_guarded_page_is_carved_when_shared() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let first = reserve(&mut pool, &mut storage, KIB);
        let _guard = pool.guard(first.descriptor().location());

        assert!(
            pool.try_reserve_sharing(KIB, PinnedRoom::Share, &mut ErrorGraph::default())
                .is_some()
        );
    }

    /// A periodic cleanup keeps every page.
    #[test_log::test]
    fn a_periodic_cleanup_keeps_every_page() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        core::mem::drop(reserve(&mut pool, &mut storage, 8 * KIB));

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Periodic,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 1);
    }

    /// An explicit cleanup returns every empty unguarded page and renumbers
    /// the rest.
    #[test_log::test]
    fn an_explicit_cleanup_returns_empty_pages() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let held: Vec<_> = (0..9)
            .map(|_| reserve(&mut pool, &mut storage, 8 * KIB))
            .collect();
        let last = held
            .last()
            .expect("nine reservations should be held")
            .clone();
        core::mem::drop(held);
        let kept = reserve(&mut pool, &mut storage, 60 * KIB);
        core::mem::drop(last);

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Explicit,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 1);
        assert_eq!(kept.descriptor().location().page, 0);
    }

    /// An explicit cleanup keeps a guarded page, empty or not.
    #[test_log::test]
    fn an_explicit_cleanup_keeps_a_guarded_page() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let first = reserve(&mut pool, &mut storage, KIB);
        let _guard = pool.guard(first.descriptor().location());
        core::mem::drop(first);

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Explicit,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 1);
    }

    /// A lazily carved page is not backed for a binding with no claim on it.
    #[test_log::test]
    fn a_lazy_page_is_not_backed_for_a_foreign_binding() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let _handle = pool
            .reserve(
                &mut storage,
                KIB,
                PageMapping::Lazy,
                &mut ErrorGraph::default(),
            )
            .expect("the pool should serve the size");

        pool.materialize(&mut storage, &ManagedMemoryHandle::new().binding())
            .expect("an unplaced binding should be left alone");

        assert_eq!(pool.report().unmapped_page_count, 1);
    }

    /// A lazily carved page is backed once a binding of it is materialized.
    #[test_log::test]
    fn a_lazy_page_is_backed_once_materialized() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let handle = pool
            .reserve(
                &mut storage,
                KIB,
                PageMapping::Lazy,
                &mut ErrorGraph::default(),
            )
            .expect("the pool should serve the size");

        pool.materialize(&mut storage, &handle.clone().binding())
            .expect("the storage should back the page");

        assert_eq!(pool.report().unmapped_page_count, 0);
    }

    /// A pool holding no page is not pinned.
    #[test_log::test]
    fn a_pool_without_pages_is_not_pinned() {
        assert!(!pool().is_pinned());
    }

    /// A pool is pinned once graphs guard every page it holds.
    #[test_log::test]
    fn a_pool_is_pinned_once_every_page_is_guarded() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let first = reserve(&mut pool, &mut storage, KIB);
        let _guard = pool.guard(first.descriptor().location());

        assert!(pool.is_pinned());
    }
}
