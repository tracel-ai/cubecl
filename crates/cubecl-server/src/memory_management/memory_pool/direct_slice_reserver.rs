use super::{
    ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice, SliceReserver, bind_slice,
    fit_whole_slice, materialize_whole, missing, padded_size, pool_origin, release_slice,
    slices_usage, whole_slice,
};
use crate::memory_management::{
    BytesFormat, Cleanup, ErrorGraph, MemoryLocation, MemoryPoolKind, MemoryPoolReport, PageGuard,
};
use crate::server::IoError;
use crate::storage::StorageAllocator;
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
pub(crate) struct DirectSliceReserver {
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
    /// A pool that serves any size up to `max_alloc_size`, the largest
    /// single allocation the device makes: with no pages to fit an allocation
    /// into, that is its only limit.
    pub(crate) fn new(
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
    pub(crate) fn reclaim<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> bool {
        let Some(ceiling) = self.reclaim_at else {
            return false;
        };
        let reserved = self.reserved();
        self.releasing.extend(free_positions(&self.slices).scan(
            reserved,
            |reserved, (index, size)| {
                (*reserved > ceiling).then(|| {
                    *reserved -= size;
                    index
                })
            },
        ));
        self.release_marked(storage, failures)
    }

    fn live(&self) -> impl Iterator<Item = &Slice> {
        self.slices.iter().flatten()
    }

    /// The pool's reserved bytes, live and free alike.
    fn reserved(&self) -> u64 {
        self.live().map(Slice::effective_size).sum()
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
            release_slice(&mut slice, storage, failures);
            self.vacant.push(index);
        }
        released
    }
}

impl SliceReserver for DirectSliceReserver {
    fn accepts_bytes(&self, size: u64) -> bool {
        padded_size(size, self.alignment).is_some_and(|padded| padded <= self.max_alloc_size)
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
        fit_whole_slice(slice, size, effective_size);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);
        Some(slice.handle.clone())
    }

    fn alloc<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        _failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let Some(effective_size) =
            padded_size(size, self.alignment).filter(|padded| *padded <= self.max_alloc_size)
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
        let slice = whole_slice(storage, size, effective_size, mapping, location)?;
        let handle = slice.handle.clone();
        place(&mut self.slices, index, slice);
        self.peak_pages = self.peak_pages.max(self.live().count() as u64);
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
        materialize_whole(slice, binding, storage)
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        let slice = self.slices.get(location.slice as usize)?.as_ref()?;
        Some(PageGuard::allocation(slice.handle.to_binding()))
    }

    fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Direct,
            usage: slices_usage(self.live()),
            page_count: self.live().count() as u64,
            peak_page_count: self.peak_pages,
            unmapped_page_count: self.live().filter(|slice| !slice.mapped).count() as u64,
            largest_alloc_bytes: self.largest_alloc_bytes,
        }
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
            .extend(free_positions(&self.slices).map(|(index, _)| index));
        self.release_marked(storage, failures);
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let slice = self.find_mut(&reserved.binding())?;
        bind_slice(slice, assigned, cursor, failures);
        Ok(())
    }
}

/// The position and padded size of every free slice among `slices`, in
/// position order.
fn free_positions(slices: &[Option<Slice>]) -> impl Iterator<Item = (usize, u64)> + '_ {
    slices.iter().enumerate().filter_map(|(index, slice)| {
        let slice = slice.as_ref().filter(|slice| slice.is_free())?;
        Some((index, slice.effective_size()))
    })
}

/// Put `slice` at `index` of `slices`, a tombstone or one past the end.
fn place(slices: &mut Vec<Option<Slice>>, index: usize, slice: Slice) {
    slices.resize_with(slices.len().max(index + 1), || None);
    slices[index] = Some(slice);
}

impl core::fmt::Display for DirectSliceReserver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let usage = slices_usage(self.live());
        if usage.reserved_bytes == 0 {
            return Ok(());
        }

        f.write_fmt(format_args!(
            "  - Direct: {} slices, largest {}\n",
            self.live().count(),
            BytesFormat::from_byte_count(self.largest_alloc_bytes)
        ))?;
        f.write_fmt(format_args!("\n{usage}\n"))
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

    /// A size past the largest allocation is refused as `BufferTooBig`
    /// before the storage is asked for anything.
    #[test_log::test]
    fn an_allocation_past_the_limit_is_refused_before_the_storage() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        let refused = alloc(&mut pool, &mut storage, 2 * MAX_ALLOC_BYTES);

        assert!(
            matches!(refused, Err(IoError::BufferTooBig { size, .. }) if size == 2 * MAX_ALLOC_BYTES),
            "got {refused:?}"
        );
        assert_eq!(storage.bytes_allocated(), 0, "the storage was never asked");
        assert!(!pool.accepts_bytes(2 * MAX_ALLOC_BYTES));
    }

    /// The padded size is what the device allocates, so it is what the limit
    /// bounds: one byte past the last aligned size under the limit is refused.
    #[test_log::test]
    fn padding_counts_against_the_limit() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        assert!(pool.accepts_bytes(MAX_ALLOC_BYTES));
        assert!(!pool.accepts_bytes(MAX_ALLOC_BYTES + 1));
        assert!(
            matches!(
                alloc(&mut pool, &mut storage, MAX_ALLOC_BYTES + 1),
                Err(IoError::BufferTooBig { .. })
            ),
            "{} pads to {} bytes, past {MAX_ALLOC_BYTES}",
            MAX_ALLOC_BYTES + 1,
            MAX_ALLOC_BYTES + ALIGNMENT_BYTES
        );
    }

    /// A size whose padding passes the largest `u64` is refused rather than
    /// wrapped to a small allocation.
    #[test_log::test]
    fn a_size_whose_padding_overflows_is_refused() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        assert!(!pool.accepts_bytes(u64::MAX));
        assert!(matches!(
            alloc(&mut pool, &mut storage, u64::MAX),
            Err(IoError::BufferTooBig { .. })
        ));
        assert!(
            pool.try_reserve(u64::MAX, &mut ErrorGraph::default())
                .is_none()
        );
    }

    /// A binding takes the cursor it is bound at, which the cross-stream
    /// waits read.
    #[test_log::test]
    fn a_binding_takes_its_cursor() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let reserved = alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit");
        let assigned = ManagedMemoryHandle::new();

        pool.bind(reserved, assigned.clone(), 7, &mut ErrorGraph::default())
            .expect("the reservation should be the pool's");

        let slice = pool
            .find(&assigned.binding())
            .expect("the binding should resolve");
        assert_eq!(slice.cursor, 7);
    }

    /// A binding of a slice the pool released is refused.
    #[test_log::test]
    fn a_binding_of_a_released_slice_is_not_found() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();
        let reserved = alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit");
        let stale = ManagedMemoryHandle::new();
        stale
            .descriptor()
            .update_location(reserved.descriptor().location());
        core::mem::drop(reserved);
        pool.reclaim(&mut storage, &mut ErrorGraph::default());

        let refused = pool.bind(
            stale,
            ManagedMemoryHandle::new(),
            0,
            &mut ErrorGraph::default(),
        );

        assert!(matches!(refused, Err(IoError::NotFound { .. })));
    }

    /// A refusal leaves the pool serving: the next allocation, exactly at the
    /// limit, is served.
    #[test_log::test]
    fn the_pool_serves_after_a_refusal() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        alloc(&mut pool, &mut storage, 4 * MAX_ALLOC_BYTES)
            .expect_err("past the limit should be refused");
        let served =
            alloc(&mut pool, &mut storage, MAX_ALLOC_BYTES).expect("exactly the limit should fit");

        assert_eq!(storage.bytes_allocated(), MAX_ALLOC_BYTES);
        assert_eq!(pool.report().usage.allocation_count, 1);
        core::mem::drop(served);
    }

    /// A freed slice of the same padded size is reused rather than allocated
    /// again.
    #[test_log::test]
    fn a_freed_slice_of_the_same_size_is_reused() {
        let mut storage = BytesStorage::default();
        let mut pool = DirectSliceReserver::new(MAX_ALLOC_BYTES, ALIGNMENT_BYTES, 0, None);
        core::mem::drop(alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit"));

        let reused = pool.try_reserve(60, &mut ErrorGraph::default());

        assert!(reused.is_some(), "60 bytes pad to the same 64");
        assert_eq!(pool.report().page_count, 1);
    }

    /// A reclaim returns free slices only past the watermark, and a pool with
    /// none keeps them for an explicit cleanup.
    #[test_log::test]
    fn a_reclaim_returns_free_slices_past_the_watermark_only() {
        let mut storage = BytesStorage::default();
        let mut kept = DirectSliceReserver::new(MAX_ALLOC_BYTES, ALIGNMENT_BYTES, 0, None);
        let mut bounded = DirectSliceReserver::new(MAX_ALLOC_BYTES, ALIGNMENT_BYTES, 0, Some(64));
        for pool in [&mut kept, &mut bounded] {
            let first = alloc(pool, &mut storage, 64).expect("64 bytes should fit");
            let second = alloc(pool, &mut storage, 64).expect("64 bytes should fit");
            core::mem::drop((first, second));
        }

        assert!(!kept.reclaim(&mut storage, &mut ErrorGraph::default()));
        assert!(bounded.reclaim(&mut storage, &mut ErrorGraph::default()));
        assert_eq!(
            bounded.report().page_count,
            1,
            "down to the 64-byte watermark"
        );
    }

    /// An explicit cleanup returns every free slice, and a periodic one none.
    #[test_log::test]
    fn only_an_explicit_cleanup_returns_free_slices() {
        let mut storage = BytesStorage::default();
        let mut pool = DirectSliceReserver::new(MAX_ALLOC_BYTES, ALIGNMENT_BYTES, 0, None);
        core::mem::drop(alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit"));

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Periodic,
            &mut ErrorGraph::default(),
        );
        assert_eq!(pool.report().page_count, 1);
        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Explicit,
            &mut ErrorGraph::default(),
        );
        assert_eq!(pool.report().page_count, 0);
    }
}
