use super::{
    ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice, SliceAllocator, SliceBinder,
    SliceReserver, missing, padded_size, padded_size_within, pool_origin, slices_usage,
    unmapped_count,
};
use crate::memory_management::{
    BytesFormat, Cleanup, ErrorGraph, MemoryLocation, MemoryPoolKind, MemoryPoolReport, PageGuard,
};
use crate::server::IoError;
use crate::storage::{StorageAllocator, StorageHandle};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;

/// A pool that does no carving: one device allocation per reservation, sized
/// to the request, reused by exact size and returned to the driver when the
/// memory [reclaims](Self::reclaim) it, down to its reclaim watermark.
///
/// The naive allocator, and the reason to want it is padding. A paged pool
/// wastes the remainder of every page it carves and a bucketed exclusive pool
/// rounds each allocation up to its bucket; this wastes only what alignment
/// demands. What it pays for that is driver traffic, which is what the other
/// pools exist to avoid.
///
/// That trade is worth making in two places. Under a process mode that discards launches
/// ([`ProcessMode`](crate::execution::ProcessMode)) the traffic is
/// free: reservations are [`PageMapping::Lazy`], so a slice nothing resolves is
/// a minted id that costs no driver call to create and none to release. And on
/// a device where the workload barely fits, the padding this removes can be the
/// difference between fitting and not.
pub struct DirectSliceReserver {
    /// Every slice owns its whole device allocation, read by the position a
    /// location carries (Position), so a freed entry is tombstoned rather
    /// than removed: a live handle's location must keep pointing at its own
    /// slice.
    slices: Vec<Option<Slice>>,
    /// The tombstones' positions, the last one freed reused first (Arrival),
    /// so a fresh slice reuses an index instead of growing `slices` for the
    /// life of the process.
    vacant: Vec<usize>,
    /// The positions a release returns, filled and drained within the one
    /// call (Scratch).
    releasing: Vec<usize>,
    alignment: u64,
    /// The largest device allocation this pool asks the storage for, padding
    /// included. A larger one is refused before the storage sees it: some
    /// backends (wgpu) treat a buffer past the device's limit as a validation
    /// error that poisons the device rather than as a failed allocation.
    max_alloc_size: u64,
    location_base: MemoryLocation,
    /// Reserved-bytes ceiling above which a [reclaim](Self::reclaim) returns
    /// free slices to the driver. A watermark, not a budget: live memory is
    /// not something this pool can decline to provide. `None` never reclaims,
    /// leaving it to an explicit cleanup.
    reclaim_at: Option<u64>,
    /// The most slices ever held at once.
    peak_pages: u64,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc_bytes: u64,
}

impl DirectSliceReserver {
    /// The pool's shape, usage and high-water marks.
    pub fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Direct,
            usage: slices_usage(self.held()),
            page_count: self.held().count() as u64,
            peak_page_count: self.peak_pages,
            unmapped_page_count: unmapped_count(self.held()),
            largest_alloc_bytes: self.largest_alloc_bytes,
        }
    }

    /// A pool that serves any size up to `max_alloc_size`, the largest
    /// single allocation the device makes: with no pages to fit an allocation
    /// into, that is its only limit.
    pub fn new(
        max_alloc_size: u64,
        alignment: u64,
        pool_index: u8,
        reclaim_at: Option<u64>,
    ) -> Self {
        Self {
            slices: Vec::new(),
            vacant: Vec::new(),
            releasing: Vec::new(),
            alignment,
            max_alloc_size,
            location_base: pool_origin(pool_index),
            reclaim_at,
            peak_pages: 0,
            largest_alloc_bytes: 0,
        }
    }

    /// Hold `storage`, an allocation the storage made outside any pool, as a slice of its own:
    /// the handle a bind assigns it from. Freed, it goes back to the storage like the pool's own.
    ///
    /// # Assumptions
    ///
    /// `storage` came from the storage this pool frees into, and no slice holds it yet.
    pub fn adopt(&mut self, storage: StorageHandle) -> ManagedMemoryHandle {
        let size = storage.size();
        let index = self.vacant.pop().unwrap_or(self.slices.len());
        let slice = Slice::new(storage, 0);
        slice.descriptor().update_location(MemoryLocation {
            slice: index as u32,
            ..self.location_base
        });
        let handle = slice.handle.clone();
        self.place(index, slice);
        self.peak_pages = self.peak_pages.max(self.held().count() as u64);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);
        handle
    }

    /// Bring the pool back under its reclaim watermark, returning free slices
    /// to the driver in position order until it fits: for a pool whose
    /// watermark is zero, every freed slice. Answers whether any went back,
    /// so the caller knows the storage has deallocations to flush.
    ///
    /// The only place slices go back outside an explicit cleanup: an
    /// allocation never releases one, so a reservation that must keep every
    /// slice where it is can still allocate. Position order is an arbitrary
    /// choice, but a deterministic one, which is what keeps a replayed
    /// allocation stream landing the same way twice.
    pub fn reclaim<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> bool {
        let Some(ceiling) = self.reclaim_at else {
            return false;
        };
        let mut excess = self.reserved().saturating_sub(ceiling);
        for (index, size) in Self::free_positions(&self.slices) {
            if excess == 0 {
                break;
            }
            excess = excess.saturating_sub(size);
            self.releasing.push(index);
        }
        self.release_marked(storage, failures)
    }
}

impl DirectSliceReserver {
    /// Every slice the pool holds, live or free.
    fn held(&self) -> impl Iterator<Item = &Slice> {
        self.slices.iter().flatten()
    }

    /// The pool's reserved bytes, live and free alike.
    fn reserved(&self) -> u64 {
        self.held().map(Slice::effective_size).sum()
    }

    /// Return every slice whose position `releasing` holds, tombstoning each
    /// index. Answers whether any went back.
    fn release_marked<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> bool {
        let released = !self.releasing.is_empty();
        for index in self.releasing.drain(..) {
            let mut slice = self.slices[index]
                .take()
                .expect("a slice marked for release should be held");
            SliceAllocator::new(storage).release_whole(&mut slice, failures);
            self.vacant.push(index);
        }
        released
    }
}

impl SliceReserver for DirectSliceReserver {
    fn accepts_bytes(&self, size: u64) -> bool {
        padded_size_within(size, self.alignment, self.max_alloc_size).is_some()
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let index = binding.descriptor().slice();
        self.slices
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| missing("slice", index))
    }

    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let index = binding.descriptor().slice();
        self.slices
            .get_mut(index)
            .and_then(Option::as_mut)
            .ok_or_else(|| missing("slice", index))
    }

    /// Reuse a freed slice of exactly this padded size. Exact-fit only: a
    /// slice handed out for a smaller request would reintroduce the padding
    /// the pool exists to remove.
    fn try_reserve(
        &mut self,
        size: u64,
        _failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        let effective_size = padded_size(size, self.alignment)?;
        let slice = self
            .slices
            .iter_mut()
            .flatten()
            .find(|slice| slice.is_free() && slice.effective_size() == effective_size)?;
        SliceBinder.fit_whole(slice, size, effective_size);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);
        Some(slice.handle.clone())
    }

    /// Allocates past what the trait assumes: a size
    /// [`accepts_bytes`](SliceReserver::accepts_bytes) refuses is answered,
    /// which the memory's dedicated and persistent windows rely on.
    ///
    /// # Guarantees
    ///
    /// A size `accepts_bytes` refuses is refused with
    /// [`IoError::BufferTooBig`] before the storage is asked.
    fn alloc<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        _failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let Some(effective_size) = padded_size_within(size, self.alignment, self.max_alloc_size)
        else {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        };
        let index = self.vacant.pop().unwrap_or(self.slices.len());
        let location = MemoryLocation {
            slice: index as u32,
            ..self.location_base
        };
        let slice =
            SliceAllocator::new(storage).allocate_whole(size, effective_size, mapping, location)?;
        let handle = slice.handle.clone();
        self.place(index, slice);
        self.peak_pages = self.peak_pages.max(self.held().count() as u64);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);
        Ok(handle)
    }

    fn materialize<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let slice = self
            .slices
            .get_mut(binding.descriptor().slice())
            .and_then(Option::as_mut);
        SliceAllocator::new(storage).materialize_whole(slice, binding)
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        let slice = self.slices.get(location.slice as usize)?.as_ref()?;
        Some(PageGuard::allocation(slice.handle.to_binding()))
    }

    /// On an explicit cleanup, return every free slice, whatever the
    /// watermark says: the caller is stating that reuse is worth less than
    /// the memory right now, which is exactly the judgement the watermark
    /// automates in the absence of one.
    ///
    /// Periodic cleanups are ignored here: the memory calls
    /// [`reclaim`](Self::reclaim) for those, which honours the watermark.
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
        self.releasing
            .extend(Self::free_positions(&self.slices).map(|(index, _)| index));
        self.release_marked(storage, failures);
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) {
        let slice = self
            .find_mut(&reserved.binding())
            .expect("a reserved slice should be held by the pool that reserved it");
        SliceBinder.bind(slice, assigned, cursor, failures);
    }
}

impl core::fmt::Display for DirectSliceReserver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let usage = slices_usage(self.held());
        if usage.reserved_bytes == 0 {
            return Ok(());
        }

        f.write_fmt(format_args!(
            "  - Direct: {} slices, largest {}\n",
            self.held().count(),
            BytesFormat::from_byte_count(self.largest_alloc_bytes)
        ))?;
        f.write_fmt(format_args!("\n{usage}\n"))
    }
}

impl DirectSliceReserver {
    /// The position and padded size of every free slice among `slices`, in
    /// position order.
    fn free_positions(slices: &[Option<Slice>]) -> impl Iterator<Item = (usize, u64)> + '_ {
        slices.iter().enumerate().filter_map(|(index, slice)| {
            let slice = slice.as_ref().filter(|slice| slice.is_free())?;
            Some((index, slice.effective_size()))
        })
    }

    /// Put `slice` at `index` of `slices`, a tombstone or one past the end.
    fn place(&mut self, index: usize, slice: Slice) {
        self.slices
            .resize_with(self.slices.len().max(index + 1), || None);
        self.slices[index] = Some(slice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{BytesStorage, StorageAllocator};

    /// The largest allocation the pools under test may make.
    const MAX_ALLOC_BYTES: u64 = 1024;
    const ALIGNMENT_BYTES: u64 = 16;

    fn pool() -> DirectSliceReserver {
        DirectSliceReserver::new(MAX_ALLOC_BYTES, ALIGNMENT_BYTES, 0, Some(0))
    }

    fn kept_pool(reclaim_at: Option<u64>) -> DirectSliceReserver {
        DirectSliceReserver::new(MAX_ALLOC_BYTES, ALIGNMENT_BYTES, 0, reclaim_at)
    }

    fn alloc(
        pool: &mut DirectSliceReserver,
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

    /// A size past the largest allocation is refused as `BufferTooBig`.
    #[test_log::test]
    fn an_allocation_past_the_limit_is_refused() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        let refused = alloc(&mut pool, &mut storage, 2 * MAX_ALLOC_BYTES);

        assert!(
            matches!(refused, Err(IoError::BufferTooBig { size, .. }) if size == 2 * MAX_ALLOC_BYTES),
            "got {refused:?}"
        );
    }

    /// A refused allocation never reaches the storage.
    #[test_log::test]
    fn a_refused_allocation_never_reaches_the_storage() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        alloc(&mut pool, &mut storage, 2 * MAX_ALLOC_BYTES)
            .expect_err("past the limit should be refused");

        assert_eq!(storage.bytes_allocated(), 0);
    }

    /// The padded size is what the device allocates, so it is what the limit
    /// bounds: one byte past the last aligned size under the limit is not
    /// accepted.
    #[test_log::test]
    fn padding_counts_against_the_limit() {
        assert!(pool().accepts_bytes(MAX_ALLOC_BYTES));
        assert!(!pool().accepts_bytes(MAX_ALLOC_BYTES + 1));
    }

    /// A size whose padding passes the largest `u64` is not accepted.
    #[test_log::test]
    fn a_size_whose_padding_overflows_is_not_accepted() {
        assert!(!pool().accepts_bytes(u64::MAX));
    }

    /// A size whose padding passes the largest `u64` is refused rather than
    /// wrapped to a small allocation.
    #[test_log::test]
    fn a_size_whose_padding_overflows_is_refused() {
        let mut storage = BytesStorage::default();

        assert!(matches!(
            alloc(&mut pool(), &mut storage, u64::MAX),
            Err(IoError::BufferTooBig { .. })
        ));
    }

    /// A size whose padding passes the largest `u64` reuses no slice.
    #[test_log::test]
    fn a_size_whose_padding_overflows_reuses_nothing() {
        assert!(
            pool()
                .try_reserve(u64::MAX, &mut ErrorGraph::default())
                .is_none()
        );
    }

    /// Binding a slice the pool released panics.
    #[test_log::test]
    #[should_panic(expected = "a reserved slice should be held by the pool that reserved it")]
    fn binding_a_released_slice_panics() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let reserved = alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit");
        let stale = ManagedMemoryHandle::new();
        stale
            .descriptor()
            .update_location(reserved.descriptor().location());
        core::mem::drop(reserved);
        pool.reclaim(&mut storage, &mut ErrorGraph::default());

        pool.bind(
            stale,
            ManagedMemoryHandle::new(),
            0,
            &mut ErrorGraph::default(),
        );
    }

    /// A refusal leaves the pool serving: the next allocation, exactly at the
    /// limit, is served.
    #[test_log::test]
    fn the_pool_serves_after_a_refusal() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        alloc(&mut pool, &mut storage, 4 * MAX_ALLOC_BYTES)
            .expect_err("past the limit should be refused");

        let _served =
            alloc(&mut pool, &mut storage, MAX_ALLOC_BYTES).expect("exactly the limit should fit");

        assert_eq!(pool.report().usage.allocation_count, 1);
    }

    /// A freed slice of the same padded size is reused.
    #[test_log::test]
    fn a_freed_slice_of_the_same_size_is_reused() {
        let mut storage = BytesStorage::default();
        let mut pool = kept_pool(None);
        core::mem::drop(alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit"));

        let reused = pool.try_reserve(60, &mut ErrorGraph::default());

        assert!(reused.is_some(), "60 bytes pad to the same 64");
    }

    /// A pool with no watermark reclaims nothing.
    #[test_log::test]
    fn a_pool_without_a_watermark_reclaims_nothing() {
        let mut storage = BytesStorage::default();
        let mut pool = kept_pool(None);
        core::mem::drop(alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit"));

        assert!(!pool.reclaim(&mut storage, &mut ErrorGraph::default()));
    }

    /// A reclaim returns free slices only down to the watermark.
    #[test_log::test]
    fn a_reclaim_returns_free_slices_down_to_the_watermark() {
        let mut storage = BytesStorage::default();
        let mut pool = kept_pool(Some(64));
        let first = alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit");
        let second = alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit");
        core::mem::drop((first, second));

        pool.reclaim(&mut storage, &mut ErrorGraph::default());

        assert_eq!(pool.report().page_count, 1, "down to the 64-byte watermark");
    }

    /// A periodic cleanup returns no free slice.
    #[test_log::test]
    fn a_periodic_cleanup_returns_nothing() {
        let mut storage = BytesStorage::default();
        let mut pool = kept_pool(None);
        core::mem::drop(alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit"));

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Periodic,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 1);
    }

    /// An explicit cleanup returns every free slice.
    #[test_log::test]
    fn an_explicit_cleanup_returns_every_free_slice() {
        let mut storage = BytesStorage::default();
        let mut pool = kept_pool(None);
        core::mem::drop(alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit"));

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Explicit,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 0);
    }

    /// A guard of a location naming no slice is refused.
    #[test_log::test]
    fn a_guard_of_no_slice_is_none() {
        assert!(pool().guard(MemoryLocation::new(0, 0, 3)).is_none());
    }
}
