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
/// That trade is worth making in two places. Under a
/// [`DryRun`](crate::dry_run::DryRun) the traffic is free: reservations are
/// [`PageMapping::Lazy`], so a slice nothing resolves is a minted id that
/// costs no driver call to create and none to release. And on a device where
/// the workload barely fits, the padding this removes can be the difference
/// between fitting and not.
pub struct DirectPool {
    /// Every slice owns its whole device allocation. Indexed by
    /// [`MemoryLocation::slice`], so freed entries are tombstoned rather than
    /// removed — a live handle's location must keep pointing at its own slice.
    slices: Vec<Option<Slice>>,
    /// Positions of the tombstones, so a fresh slice reuses an index instead
    /// of growing `slices` for the life of the process.
    vacant: Vec<usize>,
    alignment: u64,
    location_base: MemoryLocation,
    /// Reserved-bytes ceiling above which a [reclaim](Self::reclaim) returns
    /// free slices to the driver. A watermark, not a budget: live memory is
    /// not something this pool can decline to provide. `None` never reclaims,
    /// leaving it to an explicit cleanup.
    reclaim_at: Option<u64>,
    /// The most slices ever held at once.
    pages_peak: u64,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc: u64,
}

impl DirectPool {
    /// Create a pool that accepts any size: with no pages to fit an allocation
    /// into, the only limit is what the storage can allocate.
    pub fn new(alignment: u64, pool_pos: u8, reclaim_at: Option<u64>) -> Self {
        Self {
            slices: Vec::new(),
            vacant: Vec::new(),
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            reclaim_at,
            pages_peak: 0,
            largest_alloc: 0,
        }
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Direct,
            usage: self.get_memory_usage(),
            pages: self.live().count() as u64,
            pages_peak: self.pages_peak,
            pages_unmapped: self.live().filter(|slice| !slice.mapped).count() as u64,
            largest_alloc: self.largest_alloc,
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
    fn accept(&self, _size: u64) -> bool {
        true
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
        self.largest_alloc = self.largest_alloc.max(size);

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
        self.pages_peak = self.pages_peak.max(self.live().count() as u64);
        self.largest_alloc = self.largest_alloc.max(size);

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
        Some(PageGuard::allocation(slice.handle.clone().binding()))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used: Vec<_> = self.live().filter(|slice| !slice.is_free()).collect();

        MemoryUsage {
            number_allocs: used.len() as u64,
            bytes_in_use: used.iter().map(|slice| slice.storage.size()).sum(),
            bytes_padding: used.iter().map(|slice| slice.padding).sum(),
            bytes_reserved: self.live().map(|slice| slice.effective_size()).sum(),
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
        if usage.bytes_reserved == 0 {
            return Ok(());
        }

        f.write_fmt(format_args!(
            "  - Direct: {} slices, largest {}\n",
            self.live().count(),
            BytesFormat::new(self.largest_alloc)
        ))?;
        f.write_fmt(format_args!("\n{usage}\n"))
    }
}
