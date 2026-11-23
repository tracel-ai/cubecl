use super::{ManagedMemoryHandle, MemoryPool, PageMapping, Slice, calculate_padding};
use crate::memory_management::Cleanup;
use crate::memory_management::{
    BytesFormat, ErrorGraph, MemoryLocation, MemoryPoolKind, MemoryPoolReport, PageGuard,
};
use crate::storage::StorageUtilization;
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;

/// A pool that does no carving: one device allocation per reservation, sized
/// to the request, reused by exact size and returned to the driver when the
/// memory [reclaims](Self::reclaim) it, down to
/// [`reclaim_at`](Self::reclaim_at).
///
/// The naive allocator, and the reason to want it is padding. A sliced pool
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
pub struct DirectPool {
    /// Every slice owns its whole device allocation. Indexed by
    /// [`MemoryLocation::slice`], so freed entries are tombstoned rather than
    /// removed — a live handle's location must keep pointing at its own slice.
    slices: Vec<Option<Slice>>,
    /// Positions of the tombstones, so a fresh slice reuses an index instead
    /// of growing `slices` for the life of the process.
    vacant: Vec<usize>,
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

impl DirectPool {
    /// Create a pool that serves any size up to `max_alloc_size`, the largest
    /// single allocation the device makes: with no pages to fit an allocation
    /// into, that is its only limit.
    pub fn new(max_alloc_size: u64, alignment: u64, pool_pos: u8, reclaim_at: Option<u64>) -> Self {
        Self {
            slices: Vec::new(),
            vacant: Vec::new(),
            alignment,
            max_alloc_size,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            reclaim_at,
            peak_pages: 0,
            largest_alloc_bytes: 0,
        }
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Direct,
            usage: self.get_memory_usage(),
            pages: self.live().count() as u64,
            peak_pages: self.peak_pages,
            unmapped_pages: self.live().filter(|slice| !slice.mapped).count() as u64,
            largest_alloc_bytes: self.largest_alloc_bytes,
        }
    }

    fn live(&self) -> impl Iterator<Item = &Slice> {
        self.slices.iter().flatten()
    }

    /// The pool's reserved bytes, live and free alike.
    fn reserved(&self) -> u64 {
        self.live().map(|slice| slice.effective_size()).sum()
    }

    /// Return free slices to the driver until `headroom` bytes fit under
    /// [`reclaim_at`](Self::reclaim_at), tombstoning each index as it goes.
    ///
    /// Stops as soon as there is room, so an allocation that needs one slice
    /// back does not cost the reuse of every other. Slices are visited in
    /// index order — an arbitrary choice, but a deterministic one, which is
    /// what keeps a replayed allocation stream landing the same way twice. A
    /// slice that was never materialized has nothing behind its minted id, so
    /// it is dropped without troubling the driver.
    ///
    /// Answers whether any slice went back, so a caller knows the storage has
    /// deallocations to flush.
    fn release_free<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        headroom: u64,
        failures: &mut ErrorGraph,
    ) -> bool {
        let Some(ceiling) = self.reclaim_at else {
            return false;
        };
        let mut reserved = self.reserved();
        if reserved + headroom <= ceiling {
            return false;
        }
        let mut released = false;

        for (index, entry) in self.slices.iter_mut().enumerate() {
            if reserved + headroom <= ceiling {
                break;
            }
            let Some(slice) = entry else { continue };
            if !slice.is_free() {
                continue;
            }
            if slice.mapped {
                storage.dealloc(slice.storage.id);
            }
            slice.tainted.clear(failures);
            reserved -= slice.effective_size();
            *entry = None;
            self.vacant.push(index);
            released = true;
        }
        released
    }

    /// Bring the pool back under [`reclaim_at`](Self::reclaim_at): for a pool
    /// whose watermark is zero, return every freed slice. Answers whether any
    /// went back.
    ///
    /// The only place slices go back outside an explicit cleanup: an
    /// allocation never releases one, so a reservation that must keep every
    /// slice where it is can still allocate.
    pub(crate) fn reclaim<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> bool {
        self.release_free(storage, 0, failures)
    }
}

impl MemoryPool for DirectPool {
    fn accept(&self, size: u64) -> bool {
        size + calculate_padding(size, self.alignment) <= self.max_alloc_size
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let index = binding.descriptor().slice();

        self.slices
            .get(index)
            .and_then(|slice| slice.as_ref())
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {index} doesn't exist").into(),
            })
    }

    fn find_mut(&mut self, binding: &super::ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let index = binding.descriptor().slice();

        self.slices
            .get_mut(index)
            .and_then(|slice| slice.as_mut())
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {index} doesn't exist").into(),
            })
    }

    /// Reuse a freed slice of exactly this size. Exact-fit only: a slice
    /// handed out for a smaller request would reintroduce the padding the pool
    /// exists to remove.
    fn try_reserve(
        &mut self,
        size: u64,
        _failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;
        let slice = self
            .slices
            .iter_mut()
            .flatten()
            .find(|slice| slice.is_free() && slice.effective_size() == effective_size)?;

        // Both, or `effective_size()` stops describing the device allocation:
        // the slice keeps its old padding while its utilization takes the new
        // size, and the next exact-fit lookup no longer recognizes it.
        slice.padding = padding;
        slice.storage.utilization = StorageUtilization { offset: 0, size };
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);

        Some(slice.handle.clone())
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        _failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        if effective_size > self.max_alloc_size {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        }

        let storage_handle = mapping.storage_handle(storage, effective_size)?;

        let mut slice = Slice::new(storage_handle, padding);
        slice.mapped = matches!(mapping, PageMapping::Eager);
        slice.storage.utilization = StorageUtilization { offset: 0, size };

        let index = match self.vacant.pop() {
            Some(index) => index,
            None => {
                self.slices.push(None);
                self.slices.len() - 1
            }
        };
        let mut location = self.location_base;
        location.slice = index as u32;
        slice.descriptor().update_location(location);

        let handle = slice.handle.clone();
        self.slices[index] = Some(slice);
        self.peak_pages = self.peak_pages.max(self.live().count() as u64);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);

        Ok(handle)
    }

    fn materialize<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &super::ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        // An out-of-range or stale location is `find`'s error to report, not
        // ours: backing a slice this binding has no claim on would allocate
        // device memory nobody asked to resolve.
        let index = binding.descriptor().slice();
        let Some(slice) = self.slices.get_mut(index).and_then(|slice| slice.as_mut()) else {
            return Ok(());
        };
        if slice.mapped || slice.handle.descriptor() != binding.descriptor() {
            return Ok(());
        }

        slice.materialize(storage)
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        let slice = self.slices.get(location.slice as usize)?.as_ref()?;
        Some(PageGuard::allocation(slice.handle.to_binding()))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used: Vec<_> = self.live().filter(|slice| !slice.is_free()).collect();

        MemoryUsage {
            allocations: used.len() as u64,
            in_use_bytes: used.iter().map(|slice| slice.storage.size()).sum(),
            padding_bytes: used.iter().map(|slice| slice.padding).sum(),
            reserved_bytes: self.live().map(|slice| slice.effective_size()).sum(),
        }
    }

    /// On an explicit cleanup, return **every** free slice, whatever the
    /// ceiling says: the caller is stating that reuse is worth less than the
    /// memory right now, which is exactly the judgement
    /// [`reclaim_at`](Self::reclaim_at) automates in the absence of one.
    ///
    /// Periodic cleanups are ignored here: the memory calls
    /// [`reclaim`](Self::reclaim) for those, which honours the ceiling.
    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        if cleanup == Cleanup::Periodic {
            return;
        }

        for (index, entry) in self.slices.iter_mut().enumerate() {
            let Some(slice) = entry else { continue };
            if !slice.is_free() {
                continue;
            }
            if slice.mapped {
                storage.dealloc(slice.storage.id);
            }
            slice.tainted.clear(failures);
            *entry = None;
            self.vacant.push(index);
        }
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        _cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let index = reserved.descriptor().slice();
        let slice = self
            .slices
            .get_mut(index)
            .and_then(|slice| slice.as_mut())
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {index} doesn't exist").into(),
            })?;

        assigned
            .descriptor()
            .update_location(reserved.descriptor().location());
        slice.bind(assigned, failures);

        Ok(())
    }
}

impl core::fmt::Display for DirectPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let usage = self.get_memory_usage();
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
    use crate::storage::{BytesStorage, ComputeStorage};

    /// The largest allocation the pools under test may make.
    const MAX: u64 = 1024;
    const ALIGNMENT: u64 = 16;

    fn pool() -> DirectPool {
        DirectPool::new(MAX, ALIGNMENT, 0, Some(0))
    }

    fn alloc(
        pool: &mut DirectPool,
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

        let refused = alloc(&mut pool, &mut storage, 2 * MAX);

        assert!(
            matches!(refused, Err(IoError::BufferTooBig { size, .. }) if size == 2 * MAX),
            "got {refused:?}"
        );
        assert_eq!(storage.bytes_allocated(), 0, "the storage was never asked");
        assert!(!pool.accept(2 * MAX));
    }

    /// The padded size is what the device allocates, so it is what the limit
    /// bounds: one byte past the last aligned size under the limit is refused.
    #[test_log::test]
    fn padding_counts_against_the_limit() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        assert!(pool.accept(MAX));
        assert!(!pool.accept(MAX + 1));
        assert!(
            matches!(
                alloc(&mut pool, &mut storage, MAX + 1),
                Err(IoError::BufferTooBig { .. })
            ),
            "{} pads to {} bytes, past {MAX}",
            MAX + 1,
            MAX + ALIGNMENT
        );
    }

    /// A refusal leaves the pool serving: the next allocation, exactly at the
    /// limit, is served.
    #[test_log::test]
    fn the_pool_serves_after_a_refusal() {
        let mut storage = BytesStorage::default();
        let mut pool = pool();

        alloc(&mut pool, &mut storage, 4 * MAX).expect_err("past the limit");
        let served = alloc(&mut pool, &mut storage, MAX).expect("exactly the limit fits");

        assert_eq!(storage.bytes_allocated(), MAX);
        assert_eq!(pool.get_memory_usage().allocations, 1);
        core::mem::drop(served);
    }
}
