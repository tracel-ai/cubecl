use super::{ManagedMemoryHandle, MemoryPool, PageMapping, Slice, calculate_padding};
use crate::memory_management::{BytesFormat, MemoryLocation, MemoryPoolKind, MemoryPoolReport};
use crate::storage::StorageUtilization;
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;

/// A pool that does no carving: one device allocation per reservation, sized
/// to the request, reused by exact size and returned to the driver only under
/// memory pressure ([`reclaim_at`](Self::reclaim_at)).
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
    /// Reserved-bytes ceiling above which free slices are returned to the
    /// driver. A watermark, not a budget: when releasing everything free still
    /// leaves the pool above it, the allocation is served anyway — live memory
    /// is not something this pool can decline to provide. `None` never
    /// reclaims on its own, leaving it to an explicit cleanup.
    ///
    /// Demand-driven rather than prompt, which is what keeps the pool from
    /// distorting an autotune measurement: deallocating on every release would
    /// charge each benchmark iteration for driver traffic the real workload
    /// never pays, and charge candidates unequally by how much they allocate.
    /// No measurement flag, no thread-local: the policy reads the pool's own
    /// bytes.
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
    fn release_free<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        headroom: u64,
    ) {
        let Some(ceiling) = self.reclaim_at else {
            return;
        };
        let mut reserved = self.reserved();
        if reserved + headroom <= ceiling {
            return;
        }

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
            reserved -= slice.effective_size();
            *entry = None;
            self.vacant.push(index);
        }
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

    /// Reuse a freed slice of exactly this size. Exact-fit only: a slice
    /// handed out for a smaller request would reintroduce the padding the pool
    /// exists to remove.
    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
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
    ) -> Result<ManagedMemoryHandle, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        // Reclaim only if this allocation would not otherwise fit under the
        // ceiling. `try_reserve` already failed, so nothing free is the right
        // size; whatever is free here is dead weight against the ceiling.
        self.release_free(storage, effective_size);

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

    fn get_memory_usage(&self) -> MemoryUsage {
        let used: Vec<_> = self.live().filter(|slice| !slice.is_free()).collect();

        MemoryUsage {
            number_allocs: used.len() as u64,
            bytes_in_use: used.iter().map(|slice| slice.storage.size()).sum(),
            bytes_padding: used.iter().map(|slice| slice.padding).sum(),
            bytes_reserved: self.live().map(|slice| slice.effective_size()).sum(),
        }
    }

    /// Return **every** free slice, whatever the ceiling says: a cleanup is
    /// the caller stating that reuse is worth less than the memory right now,
    /// which is exactly the judgement [`reclaim_at`](Self::reclaim_at)
    /// automates in the absence of one.
    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        _explicit: bool,
    ) {
        for (index, entry) in self.slices.iter_mut().enumerate() {
            let Some(slice) = entry else { continue };
            if !slice.is_free() {
                continue;
            }
            if slice.mapped {
                storage.dealloc(slice.storage.id);
            }
            *entry = None;
            self.vacant.push(index);
        }
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        _cursor: u64,
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
        slice.handle = assigned;

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
