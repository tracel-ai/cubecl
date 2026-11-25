use super::{ManagedMemoryHandle, MemoryPool, PageMapping, Slice, padded_size};
use crate::memory_management::Cleanup;
use crate::memory_management::{
    BytesFormat, ErrorGraph, MemoryLocation, MemoryPoolKind, MemoryPoolReport, PageGuard,
};
use crate::storage::StorageUtilization;
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec;
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;

pub struct PersistentPool {
    slices: Vec<Slice>,
    sizes: HashMap<u64, Vec<usize>>,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
    /// The most slices (one device allocation each) ever held at once.
    peak_pages: u64,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc_bytes: u64,
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
                BytesFormat::from_byte_count(*size)
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
            peak_pages: 0,
            largest_alloc_bytes: 0,
        }
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Persistent,
            usage: self.get_memory_usage(),
            page_count: self.slices.len() as u64,
            peak_page_count: self.peak_pages,
            unmapped_page_count: self.slices.iter().filter(|slice| !slice.mapped).count() as u64,
            largest_alloc_bytes: self.largest_alloc_bytes,
        }
    }

    pub fn has_size(&mut self, size: u64) -> bool {
        padded_size(size, self.alignment)
            .is_some_and(|effective_size| self.sizes.contains_key(&effective_size))
    }
}

impl MemoryPool for PersistentPool {
    fn accept(&self, size: u64) -> bool {
        padded_size(size, self.alignment).is_some_and(|padded| padded <= self.max_alloc_size)
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

    fn find_mut(&mut self, binding: &super::ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let slice_index = binding.descriptor().slice();

        self.slices
            .get_mut(slice_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {} doesn't exist", slice_index).into(),
            })
    }

    fn try_reserve(
        &mut self,
        size: u64,
        _failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        let effective_size = padded_size(size, self.alignment)?;

        if let Some(positions) = self.sizes.get_mut(&effective_size) {
            for pos in positions {
                let slice = &mut self.slices[*pos];

                if slice.is_free() {
                    slice.storage.utilization.size_bytes = size;
                    slice.storage.utilization.offset_bytes = 0;
                    slice.padding = effective_size - size;
                    self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);
                    return Some(slice.handle.clone());
                }
            }
        }

        None
    }

    fn alloc<Storage: crate::storage::StorageAllocator>(
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
        let padding = effective_size - size;

        // Every persistent slice owns its whole buffer, so laziness is
        // per-slice: a minted id under a process mode that discards launches (a scratch
        // session's KV cache, for instance) costs nothing until a measurement
        // actually touches it — see [`MemoryPool::materialize`].
        let storage_handle = mapping.storage_handle(storage, effective_size)?;
        let mut slice = Slice::new(storage_handle, padding);
        slice.mapped = matches!(mapping, PageMapping::Eager);
        slice.storage.utilization = StorageUtilization {
            offset_bytes: 0,
            size_bytes: size,
        };
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
        self.peak_pages = self.peak_pages.max(self.slices.len() as u64);
        self.largest_alloc_bytes = self.largest_alloc_bytes.max(size);

        Ok(handle)
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        let slice = self.slices.get(location.slice as usize)?;
        Some(PageGuard::allocation(slice.handle.to_binding()))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .slices
            .iter()
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            allocation_count: used_slices.len() as u64,
            in_use_bytes: used_slices.iter().map(|slice| slice.storage.size()).sum(),
            padding_bytes: used_slices.iter().map(|slice| slice.padding).sum(),
            reserved_bytes: self.slices.iter().map(|slice| slice.effective_size()).sum(),
        }
    }

    fn cleanup<Storage: crate::storage::StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        if cleanup == Cleanup::Explicit {
            // We have to recompute all locations, so it's just safer to rebuild everything.
            let mut slices = Vec::new();
            let mut sizes = HashMap::<u64, Vec<usize>>::new();

            for mut slice in self.slices.drain(..) {
                if slice.is_free() {
                    // A dropped slice releases the failure it carried. A
                    // minted-but-never-materialized id has nothing behind
                    // it for the driver to free.
                    slice.tainted.clear(failures);
                    if slice.mapped {
                        storage.deallocate(slice.storage.id);
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
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let slice = &mut self.slices[old.descriptor().slice()];
        new.descriptor()
            .update_location(old.descriptor().location());
        slice.cursor = cursor;
        slice.bind(new, failures);

        Ok(())
    }

    fn materialize<Storage: crate::storage::StorageAllocator>(
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

        slice.materialize(storage)
    }
}

#[cfg(test)]
mod tests {
    use crate::memory_management::memory_pool::calculate_padding;
    use crate::storage::BytesStorage;

    use super::*;

    /// A size whose padded allocation passes the largest allocation is refused
    /// as `BufferTooBig` before the storage is asked, and the pool still
    /// serves the next one that fits.
    #[test_log::test]
    fn an_allocation_past_the_limit_is_refused_before_the_storage() {
        use crate::storage::StorageAllocator;

        let mut storage = BytesStorage::default();
        let mut pool = PersistentPool::new(1024, 16, 0);

        let refused = pool.alloc(
            &mut storage,
            1025,
            PageMapping::Eager,
            &mut ErrorGraph::default(),
        );
        assert!(
            matches!(refused, Err(IoError::BufferTooBig { size: 1025, .. })),
            "got {refused:?}"
        );
        assert_eq!(storage.bytes_allocated(), 0, "the storage was never asked");

        let served = pool
            .alloc(
                &mut storage,
                1024,
                PageMapping::Eager,
                &mut ErrorGraph::default(),
            )
            .expect("exactly the limit fits");
        assert_eq!(storage.bytes_allocated(), 1024);
        core::mem::drop(served);
    }

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
            .alloc(
                &mut storage,
                size,
                PageMapping::Eager,
                &mut ErrorGraph::default(),
            )
            .expect("alloc");
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
        let mut pool = PersistentPool::new(1024, 32, 0);
        let first = pool
            .alloc(
                &mut storage,
                100,
                PageMapping::Eager,
                &mut ErrorGraph::default(),
            )
            .expect("100 bytes fit");
        core::mem::drop(first);

        let _reused = pool
            .try_reserve(97, &mut ErrorGraph::default())
            .expect("97 bytes pad to the same 128");

        let usage = pool.get_memory_usage();
        assert_eq!((usage.reserved_bytes, usage.padding_bytes), (128, 31));
    }

    /// A size whose padding passes the largest `u64` is refused rather than
    /// wrapped to a small allocation.
    #[test_log::test]
    fn a_size_whose_padding_overflows_is_refused() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentPool::new(1024, 32, 0);

        assert!(!pool.accept(u64::MAX));
        assert!(!pool.has_size(u64::MAX));
        assert!(matches!(
            pool.alloc(
                &mut storage,
                u64::MAX,
                PageMapping::Eager,
                &mut ErrorGraph::default()
            ),
            Err(IoError::BufferTooBig { .. })
        ));
    }

    #[test_log::test]
    fn persistent_pool() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentPool::new(1024 * 1024, 4, 0);

        let result = pool.try_reserve(1024, &mut ErrorGraph::default());
        assert!(result.is_none(), "No alloc yet");

        let alloc1 = pool.alloc(
            &mut storage,
            1024,
            PageMapping::Eager,
            &mut ErrorGraph::default(),
        );
        let result = pool.try_reserve(1024, &mut ErrorGraph::default());
        assert!(result.is_none(), "No free slice yet, handle1 is alive");

        core::mem::drop(alloc1);
        let result = pool.try_reserve(1024, &mut ErrorGraph::default());
        assert!(result.is_some(), "Handle1 is free to be reused.");
        core::mem::drop(result);

        let result = pool.try_reserve(1025, &mut ErrorGraph::default());
        assert!(result.is_none(), "Not the same size.");

        let alloc2 = pool.alloc(
            &mut storage,
            1024,
            PageMapping::Eager,
            &mut ErrorGraph::default(),
        );
        let usage = pool.get_memory_usage();
        assert_eq!(usage.in_use_bytes, 1024);
        assert_eq!(usage.reserved_bytes, 2048);

        let result = pool.try_reserve(1024, &mut ErrorGraph::default());
        let usage = pool.get_memory_usage();
        assert!(result.is_some(), "Handle1 is free to be reused.");
        assert_eq!(usage.in_use_bytes, 2048);
        assert_eq!(usage.reserved_bytes, 2048);

        core::mem::drop(alloc2);
        core::mem::drop(result);

        let usage = pool.get_memory_usage();
        assert_eq!(usage.in_use_bytes, 0);
        assert_eq!(usage.reserved_bytes, 2048);
    }
}
