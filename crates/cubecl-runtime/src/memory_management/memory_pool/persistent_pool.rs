use super::{
    ManagedMemoryHandle, ManagedMemoryId, MemoryPool, PageMapping, Slice, calculate_padding,
};
use crate::memory_management::{BytesFormat, MemoryLocation, MemoryPoolKind, MemoryPoolReport};
use crate::storage::StorageUtilization;
use crate::storage::{StorageHandle, StorageId};
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec;
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::{HashMap, HashSet};

pub struct PersistentPool {
    slices: Vec<Slice>,
    sizes: HashMap<u64, Vec<usize>>,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
    /// The most slices (one device allocation each) ever held at once.
    pages_peak: u64,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc: u64,
}

impl core::fmt::Display for PersistentPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (size, positions) in self.sizes.iter() {
            let mut num_free = 0;
            let mut num_full = 0;
            let total = positions.len();

            for pos in positions {
                let slice = &self.slices[*pos];
                let is_free = slice.is_free();
                if is_free {
                    num_free += 1;
                } else {
                    num_full += 1;
                }
            }

            f.write_fmt(format_args!(
                "  - Slices {} =>  {num_free} free - {num_full} full - {total} total\n",
                BytesFormat::new(*size)
            ))?;
        }

        if !self.sizes.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;
        }

        Ok(())
    }
}

impl PersistentPool {
    pub fn new(max_alloc_size: u64, alignment: u64, pool_pos: u8) -> Self {
        Self {
            slices: Vec::new(),
            sizes: HashMap::new(),
            max_alloc_size,
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            pages_peak: 0,
            largest_alloc: 0,
        }
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Persistent,
            usage: self.get_memory_usage(),
            pages: self.slices.len() as u64,
            pages_peak: self.pages_peak,
            pages_unmapped: self.slices.iter().filter(|slice| !slice.mapped).count() as u64,
            largest_alloc: self.largest_alloc,
        }
    }

    pub fn has_size(&mut self, size: u64) -> bool {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;
        self.sizes.contains_key(&effective_size)
    }

    /// Retain a handle to every slice a capture window `touched` (reserved or
    /// allocated while it was open), keeping those slices from ever being
    /// reported free (and thus reused). These are exactly the slices the graph's
    /// recorded kernels may replay against, so retaining them — and nothing more
    /// — pins precisely the graph's working set: a slice the window never touched
    /// is not retained (no over-retention), and a slice that was live at the
    /// start but was freed and reused mid-window *is* (it was touched). The graph
    /// holds the handles and releases the slices by dropping them. Cloning a
    /// slice's handle is exactly what [`try_reserve`](Self::try_reserve) does.
    pub fn retain_touched(&self, touched: &HashSet<ManagedMemoryId>) -> Vec<ManagedMemoryHandle> {
        self.slices
            .iter()
            .filter(|slice| touched.contains(&slice.descriptor().id))
            .map(|slice| slice.handle.clone())
            .collect()
    }
}

impl MemoryPool for PersistentPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let slice_index = binding.descriptor().slice();

        self.slices
            .get(slice_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {} doesn't exist", slice_index).into(),
            })
    }

    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        if let Some(positions) = self.sizes.get_mut(&effective_size) {
            for pos in positions {
                let slice = &mut self.slices[*pos];

                if slice.is_free() {
                    slice.storage.utilization.size = size;
                    slice.storage.utilization.offset = 0;
                    self.largest_alloc = self.largest_alloc.max(size);
                    return Some(slice.handle.clone());
                }
            }
        }

        None
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        // Every persistent slice owns its whole buffer, so laziness is
        // per-slice: a minted id under a dry run (a scratch session's KV
        // cache, for instance) costs nothing until a measurement actually
        // touches it — see [`MemoryPool::materialize`].
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
        let slice_id = slice.descriptor();
        let slice_pos = self.slices.len();
        let mut location = self.location_base;
        location.slice = slice_pos as u32;
        slice_id.update_location(location);

        match self.sizes.get_mut(&effective_size) {
            Some(vals) => {
                vals.push(slice_pos);
            }
            None => {
                self.sizes.insert(effective_size, vec![slice_pos]);
            }
        }

        let handle = slice.handle.clone();
        self.slices.push(slice);
        self.pages_peak = self.pages_peak.max(self.slices.len() as u64);
        self.largest_alloc = self.largest_alloc.max(size);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .slices
            .iter()
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|slice| slice.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|slice| slice.padding).sum(),
            bytes_reserved: self.slices.iter().map(|slice| slice.effective_size()).sum(),
        }
    }

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {
            // We have to recompute all locations, so it's just safer to rebuild everything.
            let mut slices = Vec::new();
            let mut sizes = HashMap::<u64, Vec<usize>>::new();

            for slice in self.slices.drain(..) {
                if slice.is_free() {
                    // A minted-but-never-materialized id has nothing behind
                    // it for the driver to free.
                    if slice.mapped {
                        storage.dealloc(slice.storage.id);
                    }
                } else {
                    let slice_pos = slices.len();
                    let effective_size = slice.effective_size();
                    slice.descriptor().update_slice(slice_pos as u32);
                    slices.push(slice);

                    match sizes.get_mut(&effective_size) {
                        Some(vals) => {
                            vals.push(slice_pos);
                        }
                        None => {
                            sizes.insert(effective_size, vec![slice_pos]);
                        }
                    }
                }
            }

            self.sizes = sizes;
            self.slices = slices;
            storage.flush();
        }
    }

    fn bind(
        &mut self,
        old: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let slice = &mut self.slices[old.descriptor().slice()];
        new.descriptor()
            .update_location(old.descriptor().location());
        slice.cursor = cursor;
        slice.handle = new;

        Ok(())
    }

    fn materialize<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &super::ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        // An out-of-range slice is `find`'s error to report, not ours. So is a
        // stale location whose index a later cleanup reassigned: without the
        // identity check it names a live slice this binding has no claim on,
        // and backing that slice would allocate device memory for an
        // allocation nobody asked to resolve.
        let Some(slice) = self.slices.get_mut(binding.descriptor().slice()) else {
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
        let slice = &mut self.slices[binding.descriptor().slice()];
        slice.storage.id = real.id;
        slice.mapped = true;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::memory_management::memory_pool::calculate_padding;
    use crate::storage::BytesStorage;

    use super::*;

    /// `alloc` and `try_reserve` must use the same `sizes` key (`size + padding`). See #1224.
    #[test_log::test]
    fn persistent_pool_try_reserve_reuses_slice_with_padding() {
        let mut storage = BytesStorage::default();
        let alignment = 4u64;
        let mut pool = PersistentPool::new(1024 * 1024, alignment, 0);

        let size = 1025u64;
        assert_ne!(
            calculate_padding(size, alignment),
            0,
            "test needs non-zero padding so alloc vs try_reserve keys differed pre-fix"
        );

        let handle = pool
            .alloc(&mut storage, size, PageMapping::Eager)
            .expect("alloc");
        assert!(
            pool.try_reserve(size).is_none(),
            "slice must stay reserved while the handle is alive"
        );

        core::mem::drop(handle);

        assert!(
            pool.try_reserve(size).is_some(),
            "freed slice should be reusable"
        );
    }

    #[test_log::test]
    fn persistent_pool() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentPool::new(1024 * 1024, 4, 0);

        let result = pool.try_reserve(1024);
        assert!(result.is_none(), "No alloc yet");

        let alloc1 = pool.alloc(&mut storage, 1024, PageMapping::Eager);
        let result = pool.try_reserve(1024);
        assert!(result.is_none(), "No free slice yet, handle1 is alive");

        core::mem::drop(alloc1);
        let result = pool.try_reserve(1024);
        assert!(result.is_some(), "Handle1 is free to be reused.");
        core::mem::drop(result);

        let result = pool.try_reserve(1025);
        assert!(result.is_none(), "Not the same size.");

        let alloc2 = pool.alloc(&mut storage, 1024, PageMapping::Eager);
        let usage = pool.get_memory_usage();
        assert_eq!(usage.bytes_in_use, 1024);
        assert_eq!(usage.bytes_reserved, 2048);

        let result = pool.try_reserve(1024);
        let usage = pool.get_memory_usage();
        assert!(result.is_some(), "Handle1 is free to be reused.");
        assert_eq!(usage.bytes_in_use, 2048);
        assert_eq!(usage.bytes_reserved, 2048);

        core::mem::drop(alloc2);
        core::mem::drop(result);

        let usage = pool.get_memory_usage();
        assert_eq!(usage.bytes_in_use, 0);
        assert_eq!(usage.bytes_reserved, 2048);
    }
}
