use super::{ManagedMemoryHandle, MemoryPool, PageMapping, Slice, calculate_padding};
use crate::memory_management::{BytesFormat, MemoryLocation, MemoryPoolKind, MemoryPoolReport};
use crate::storage::{StorageHandle, StorageId, StorageUtilization};
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;

/// A pool that does no pooling: one device allocation per reservation, sized
/// to the request, returned to the driver as soon as it is free.
///
/// The naive allocator, and the reason to want it is padding. A sliced pool
/// wastes the remainder of every page it carves and a bucketed exclusive pool
/// rounds each allocation up to its bucket; this wastes only what alignment
/// demands. What it pays for that is driver traffic — an allocation and a free
/// per reservation, which is exactly what the other pools exist to avoid.
///
/// That trade is worth making in two places. Under a
/// [`DryRun`](crate::dry_run::DryRun) the traffic is free: reservations are
/// [`PageMapping::Lazy`], so a slice nothing resolves is a minted id that
/// costs no driver call to create and none to release. And on a device where
/// the workload barely fits, the padding this removes can be the difference
/// between fitting and not.
///
/// # Measurement
///
/// Freeing on release would bias an autotune benchmark: its iterations would
/// each pay for a driver allocation the real workload does not, and pay
/// unequally across candidates that allocate differently. So while a
/// measurement is open ([`dry_run::measuring`](crate::dry_run::measuring)) the
/// pool keeps freed slices and reuses them by exact size, making the timed
/// loop allocation-free. The parked slices go back to the driver at the first
/// reservation after the measurement ends.
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
    /// The most slices ever held at once.
    pages_peak: u64,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc: u64,
}

impl DirectPool {
    /// Create a pool that accepts any size: with no pages to fit an allocation
    /// into, the only limit is what the storage can allocate.
    pub fn new(alignment: u64, pool_pos: u8) -> Self {
        Self {
            slices: Vec::new(),
            vacant: Vec::new(),
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
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

    /// Return every free slice to the driver and tombstone its index.
    ///
    /// The whole point of the pool, so it runs on the ordinary path — before
    /// each new allocation — rather than waiting for an explicit cleanup. A
    /// slice that was never materialized has nothing behind its minted id, so
    /// it is dropped without troubling the driver.
    fn release_free<Storage: crate::storage::ComputeStorage>(&mut self, storage: &mut Storage) {
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

    /// Reuse a freed slice **only while a measurement is open** — see the type
    /// doc. Outside one, returning `None` is what sends every reservation
    /// through [`alloc`](Self::alloc), which is where freed slices go back to
    /// the driver.
    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        if !crate::dry_run::measuring() {
            return None;
        }

        let effective_size = size + calculate_padding(size, self.alignment);
        let slice = self
            .slices
            .iter_mut()
            .flatten()
            .find(|slice| slice.is_free() && slice.effective_size() == effective_size)?;

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
        // Everything free goes back now. Skipped mid-measurement so a timed
        // loop that outgrows its reused slices does not start paying for
        // frees partway through, which would bias the samples after it.
        if !crate::dry_run::measuring() {
            self.release_free(storage);
        }

        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        let storage_handle = match mapping {
            PageMapping::Eager => storage.alloc(effective_size)?,
            PageMapping::Lazy => StorageHandle::new(
                StorageId::new(),
                StorageUtilization {
                    offset: 0,
                    size: effective_size,
                },
            ),
        };

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

        let effective_size = slice.effective_size();
        let real = storage
            .alloc(effective_size)
            .map_err(|err| IoError::StorageMappingFailed {
                size: effective_size,
                source: alloc::boxed::Box::new(err),
                backtrace: BackTrace::capture(),
            })?;

        let slice = self.slices[index].as_mut().expect("just checked");
        slice.storage.id = real.id;
        slice.mapped = true;

        Ok(())
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

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        _explicit: bool,
    ) {
        // No `explicit` gate and no period: releasing free slices is this
        // pool's normal behavior, not a reclamation it has to be asked for.
        self.release_free(storage);
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
