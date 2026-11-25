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
use cubecl_environment::collections::HashMap;

/// A pool of one device allocation per buffer, reused only by a buffer of
/// the same padded size: what weights and other allocations kept for a
/// model's life are served from, so they never hold a page of the dynamic
/// pools.
pub(crate) struct PersistentSliceReserver {
    /// Every slice, read by the position a location carries (Position) and
    /// added at the end (Arrival).
    slices: Vec<Slice>,
    /// The positions of the slices of each padded size (Keyed), each list in
    /// the order its slices were allocated (Arrival).
    sizes: HashMap<u64, Vec<usize>>,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
    /// The most slices (one device allocation each) ever held at once.
    peak_pages: u64,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc_bytes: u64,
}

impl PersistentSliceReserver {
    /// A pool that serves any size up to `max_alloc_size`, the largest single
    /// allocation the device makes.
    pub(crate) fn new(max_alloc_size: u64, alignment: u64, pool_index: u8) -> Self {
        Self {
            slices: Vec::new(),
            sizes: HashMap::new(),
            max_alloc_size,
            alignment,
            location_base: pool_origin(pool_index),
            peak_pages: 0,
            largest_alloc_bytes: 0,
        }
    }

    /// Whether the pool holds a slice of the padded size `size_bytes` takes,
    /// free or not.
    pub(crate) fn has_size_bytes(&self, size_bytes: u64) -> bool {
        padded_size(size_bytes, self.alignment)
            .is_some_and(|effective_size| self.sizes.contains_key(&effective_size))
    }
}

impl SliceReserver for PersistentSliceReserver {
    fn accepts_bytes(&self, size: u64) -> bool {
        padded_size(size, self.alignment).is_some_and(|padded| padded <= self.max_alloc_size)
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let index = binding.descriptor().slice();
        self.slices
            .get(index)
            .ok_or_else(|| missing("slice", index))
    }

    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let index = binding.descriptor().slice();
        self.slices
            .get_mut(index)
            .ok_or_else(|| missing("slice", index))
    }

    fn try_reserve(
        &mut self,
        size: u64,
        _failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        let effective_size = padded_size(size, self.alignment)?;
        let position = *self
            .sizes
            .get(&effective_size)?
            .iter()
            .find(|position| self.slices[**position].is_free())?;
        let slice = &mut self.slices[position];
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
        // Refused before the storage sees it: some backends (wgpu) treat a
        // buffer past the device's limit as a validation error that poisons
        // the device rather than as a failed allocation.
        let Some(effective_size) =
            padded_size(size, self.alignment).filter(|padded| *padded <= self.max_alloc_size)
        else {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        };

        // Every persistent slice owns its whole buffer, so laziness is
        // per-slice: a minted id under a process mode that discards launches
        // (a scratch session's KV cache, for instance) costs nothing until a
        // measurement actually touches it.
        let position = self.slices.len();
        let location = MemoryLocation {
            slice: position as u32,
            ..self.location_base
        };
        let slice = whole_slice(storage, size, effective_size, mapping, location)?;
        let handle = slice.handle.clone();
        self.sizes.entry(effective_size).or_default().push(position);
        self.slices.push(slice);
        self.peak_pages = self.peak_pages.max(self.slices.len() as u64);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);

        Ok(handle)
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        let slice = self.slices.get(location.slice as usize)?;
        Some(PageGuard::allocation(slice.handle.to_binding()))
    }

    fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Persistent,
            usage: slices_usage(self.slices.iter()),
            page_count: self.slices.len() as u64,
            peak_page_count: self.peak_pages,
            unmapped_page_count: self.slices.iter().filter(|slice| !slice.mapped).count() as u64,
            largest_alloc_bytes: self.largest_alloc_bytes,
        }
    }

    /// On an explicit cleanup, return every free slice and renumber the rest;
    /// a periodic one keeps everything, since weights are reused across the
    /// whole life of a model.
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
        self.slices
            .retain_mut(|slice| keep_live(slice, storage, failures));
        self.sizes.clear();
        for (position, slice) in self.slices.iter().enumerate() {
            slice.descriptor().update_slice(position as u32);
            self.sizes
                .entry(slice.effective_size())
                .or_default()
                .push(position);
        }
        storage.flush();
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let slice = self
            .slices
            .get_mut(reserved.descriptor().slice())
            .expect("a reserved slice should be held by the pool that reserved it");
        bind_slice(slice, assigned, cursor, failures);
        Ok(())
    }

    /// A stale location whose index a later cleanup reassigned names a live
    /// slice this binding has no claim on, so it is left alone too.
    fn materialize<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let slice = self.slices.get_mut(binding.descriptor().slice());
        materialize_whole(slice, binding, storage)
    }
}

/// Whether `slice` stays, releasing it when it is free.
fn keep_live<Storage: StorageAllocator>(
    slice: &mut Slice,
    storage: &mut Storage,
    failures: &mut ErrorGraph,
) -> bool {
    if !slice.is_free() {
        return true;
    }
    release_slice(slice, storage, failures);
    false
}

impl core::fmt::Display for PersistentSliceReserver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (size, positions) in self.sizes.iter() {
            let free_count = positions
                .iter()
                .filter(|position| self.slices[**position].is_free())
                .count();
            let total_count = positions.len();
            f.write_fmt(format_args!(
                "  - Slices {} =>  {free_count} free - {} full - {total_count} total\n",
                BytesFormat::from_byte_count(*size),
                total_count - free_count,
            ))?;
        }

        if !self.sizes.is_empty() {
            f.write_fmt(format_args!("\n{}\n", slices_usage(self.slices.iter())))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::memory_pool::calculate_padding;
    use crate::storage::{BytesStorage, StorageAllocator};

    fn alloc(
        pool: &mut PersistentSliceReserver,
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

    /// A size whose padded allocation passes the largest allocation is refused
    /// as `BufferTooBig` before the storage is asked, and the pool still
    /// serves the next one that fits.
    #[test_log::test]
    fn an_allocation_past_the_limit_is_refused_before_the_storage() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentSliceReserver::new(1024, 16, 0);

        let refused = alloc(&mut pool, &mut storage, 1025);
        assert!(
            matches!(refused, Err(IoError::BufferTooBig { size: 1025, .. })),
            "got {refused:?}"
        );
        assert_eq!(storage.bytes_allocated(), 0, "the storage was never asked");

        let served = alloc(&mut pool, &mut storage, 1024).expect("exactly the limit should fit");
        assert_eq!(storage.bytes_allocated(), 1024);
        core::mem::drop(served);
    }

    /// `alloc` and `try_reserve` must use the same `sizes` key (`size + padding`). See #1224.
    #[test_log::test]
    fn persistent_pool_try_reserve_reuses_slice_with_padding() {
        let mut storage = BytesStorage::default();
        let alignment = 4u64;
        let mut pool = PersistentSliceReserver::new(1024 * 1024, alignment, 0);

        let size = 1025u64;
        assert_ne!(
            calculate_padding(size, alignment),
            0,
            "test needs non-zero padding so alloc vs try_reserve keys differed pre-fix"
        );

        let handle = alloc(&mut pool, &mut storage, size).expect("the size should fit");
        assert!(
            pool.try_reserve(size, &mut ErrorGraph::default()).is_none(),
            "slice must stay reserved while the handle is alive"
        );

        core::mem::drop(handle);

        assert!(
            pool.try_reserve(size, &mut ErrorGraph::default()).is_some(),
            "freed slice should be reusable"
        );
    }

    /// A reused slice takes the padding of the size it now serves, so its
    /// padded size stays the bucket it is filed under.
    #[test_log::test]
    fn a_reused_slice_takes_the_padding_of_its_new_size() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentSliceReserver::new(1024, 32, 0);
        let first = alloc(&mut pool, &mut storage, 100).expect("100 bytes should fit");
        core::mem::drop(first);

        let _reused = pool
            .try_reserve(97, &mut ErrorGraph::default())
            .expect("97 bytes should pad to the same 128");

        let usage = pool.report().usage;
        assert_eq!((usage.reserved_bytes, usage.padding_bytes), (128, 31));
    }

    /// A size whose padding passes the largest `u64` is refused rather than
    /// wrapped to a small allocation.
    #[test_log::test]
    fn a_size_whose_padding_overflows_is_refused() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentSliceReserver::new(1024, 32, 0);

        assert!(!pool.accepts_bytes(u64::MAX));
        assert!(!pool.has_size_bytes(u64::MAX));
        assert!(matches!(
            alloc(&mut pool, &mut storage, u64::MAX),
            Err(IoError::BufferTooBig { .. })
        ));
    }

    /// A size of another padded size is not served from a freed slice.
    #[test_log::test]
    fn a_freed_slice_serves_only_its_own_padded_size() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentSliceReserver::new(1024 * 1024, 4, 0);
        core::mem::drop(alloc(&mut pool, &mut storage, 1024).expect("1024 bytes should fit"));

        assert!(pool.try_reserve(1025, &mut ErrorGraph::default()).is_none());
        assert!(pool.has_size_bytes(1024));
    }

    /// A pool's usage counts what is live and what it holds.
    #[test_log::test]
    fn the_usage_counts_live_and_held_bytes() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentSliceReserver::new(1024 * 1024, 4, 0);
        let first = alloc(&mut pool, &mut storage, 1024).expect("1024 bytes should fit");
        let _second = alloc(&mut pool, &mut storage, 1024).expect("1024 bytes should fit");
        core::mem::drop(first);

        let usage = pool.report().usage;
        assert_eq!((usage.in_use_bytes, usage.reserved_bytes), (1024, 2048));
    }

    /// An explicit cleanup returns every free slice and renumbers the live
    /// ones, which still resolve; a periodic one keeps everything.
    #[test_log::test]
    fn an_explicit_cleanup_returns_free_slices_and_renumbers_the_rest() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentSliceReserver::new(1024 * 1024, 4, 0);
        let freed = alloc(&mut pool, &mut storage, 64).expect("64 bytes should fit");
        let kept = alloc(&mut pool, &mut storage, 128).expect("128 bytes should fit");
        core::mem::drop(freed);

        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Periodic,
            &mut ErrorGraph::default(),
        );
        assert_eq!(pool.report().page_count, 2);
        pool.cleanup(
            &mut storage,
            0,
            Cleanup::Explicit,
            &mut ErrorGraph::default(),
        );

        assert_eq!(pool.report().page_count, 1);
        assert_eq!(kept.descriptor().location().slice, 0);
        assert!(pool.find(&kept.binding()).is_ok());
    }
}
