use super::{MemoryPool, RingBuffer, Slice, SliceBinding, SliceHandle, SliceId};
use crate::memory_management::{BytesFormat, MemoryUsage};
use crate::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use crate::{memory_management::memory_pool::calculate_padding, server::IoError};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// A memory pool that allocates buffers in a range of sizes and reuses them to minimize allocations.
///
/// - Each 'page' allocation will contain a number of sub slices.
/// - The pool uses a ring buffer to efficiently manage and reuse pages.
pub(crate) struct SlicedPool {
    pages: HashMap<StorageId, MemoryPage>,
    slices: HashMap<SliceId, Slice>,
    ring: RingBuffer,
    recently_added_pages: Vec<StorageId>,
    recently_allocated_size: u64,
    page_size: u64,
    max_alloc_size: u64,
    alignment: u64,
}

impl core::fmt::Display for SlicedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}\n",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.max_alloc_size)
        ))?;

        for (id, page) in self.pages.iter() {
            let num_slices = page.slices.len();
            f.write_fmt(format_args!("   - Page {id} num_slices={num_slices} =>"))?;

            let mut size_free = 0;
            let mut size_full = 0;
            let mut size_total = 0;

            for id in page.slices.values() {
                let slice = self.slices.get(id).unwrap();
                let is_free = slice.is_free();
                if is_free {
                    size_free += slice.effective_size();
                } else {
                    size_full += slice.effective_size();
                }
                size_total += slice.effective_size();
            }

            let size_free = BytesFormat::new(size_free);
            let size_full = BytesFormat::new(size_full);
            let size_total = BytesFormat::new(size_total);

            f.write_fmt(format_args!(
                " {size_free} free - {size_full} full - {size_total} total\n"
            ))?;
        }

        if !self.pages.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;
        }

        Ok(())
    }
}

// TODO: consider using generic trait and decouple from Slice
#[derive(new, Debug)]
pub(crate) struct MemoryPage {
    pub(crate) slices: HashMap<u64, SliceId>,
}

impl MemoryPage {
    /// merge slice at first_slice_address with the next slice (if there is one and if it's free)
    /// return a boolean representing if a merge happened
    pub(crate) fn merge_with_next_slice(
        &mut self,
        first_slice_address: u64,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> bool {
        let first_slice_id = self.find_slice(first_slice_address).expect(
            "merge_with_next_slice shouldn't be called with a nonexistent first_slice address",
        );

        let next_slice_address =
            first_slice_address + slices.get(&first_slice_id).unwrap().effective_size();

        if let Some(next_slice_id) = self.find_slice(next_slice_address) {
            let (next_slice_eff_size, next_slice_is_free) = {
                let next_slice = slices.get(&next_slice_id).unwrap();
                (next_slice.effective_size(), next_slice.is_free())
            };
            if next_slice_is_free {
                let first_slice = slices.get_mut(&first_slice_id).unwrap();
                let first_slice_eff_size = first_slice.effective_size();
                let first_slice_offset = first_slice.storage.offset();

                let merged_size = first_slice_eff_size + next_slice_eff_size;
                first_slice.storage.utilization = StorageUtilization {
                    size: merged_size,
                    offset: first_slice_offset,
                };
                first_slice.padding = 0;

                // Cleanup of the extra slice
                self.slices.remove(&next_slice_address);
                slices.remove(&next_slice_id);
                return true;
            }
            return false;
        }
        false
    }

    pub(crate) fn find_slice(&self, address: u64) -> Option<SliceId> {
        let slice_id = self.slices.get(&address);
        slice_id.copied()
    }

    pub(crate) fn insert_slice(&mut self, address: u64, slice: SliceId) {
        self.slices.insert(address, slice);
    }
}

impl MemoryPool for SlicedPool {
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    /// Returns the resource from the storage, for the specified handle.
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices.get(binding.id()).map(|s| &s.storage)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;
        let slice_id =
            self.ring
                .find_free_slice(effective_size, &mut self.pages, &mut self.slices)?;

        let slice = self.slices.get_mut(&slice_id).unwrap();
        let old_slice_size = slice.effective_size();
        let offset = slice.storage.utilization.offset;
        slice.storage.utilization = StorageUtilization { offset, size };
        let new_padding = old_slice_size - size;
        slice.padding = new_padding;
        assert_eq!(
            slice.effective_size(),
            old_slice_size,
            "new and old slice should have the same size"
        );

        Some(slice.handle.clone())
    }

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        let storage_id = self.create_page(storage)?;
        self.recently_added_pages.push(storage_id);
        self.recently_allocated_size += self.page_size;

        let slice = self.create_slice(0, size, storage_id);

        let effective_size = slice.effective_size();

        let extra_slice = if effective_size < self.page_size {
            Some(self.create_slice(effective_size, self.page_size - effective_size, storage_id))
        } else {
            None
        };

        let handle_slice = slice.handle.clone();
        let storage_id = slice.storage.id;
        let slice_id = slice.id();
        let slice_offset = slice.storage.offset();

        self.slices.insert(slice_id, slice);
        let page = self.pages.get_mut(&storage_id).unwrap();
        page.slices.insert(slice_offset, slice_id);

        if let Some(extra_slice) = extra_slice {
            let extra_slice_id = extra_slice.id();
            let extra_slice_offset = extra_slice.storage.offset();
            self.slices.insert(extra_slice_id, extra_slice);
            page.slices.insert(extra_slice_offset, extra_slice_id);
        }

        Ok(handle_slice)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .slices
            .values()
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|s| s.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|s| s.padding).sum(),
            bytes_reserved: (self.pages.len() as u64) * self.page_size,
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        _storage: &mut Storage,
        _alloc_nr: u64,
        _explicit: bool,
    ) {
        // This pool doesn't do any shrinking currently.
    }
}

impl SlicedPool {
    pub(crate) fn new(page_size: u64, max_alloc_size: u64, alignment: u64) -> Self {
        // Pages should be allocated to be aligned.
        assert_eq!(page_size % alignment, 0);
        Self {
            pages: HashMap::new(),
            slices: HashMap::new(),
            ring: RingBuffer::new(alignment),
            recently_added_pages: Vec::new(),
            recently_allocated_size: 0,
            alignment,
            page_size,
            max_alloc_size,
        }
    }

    /// Creates a slice of size `size` upon the given page with the given offset.
    fn create_slice(&self, offset: u64, size: u64, storage_id: StorageId) -> Slice {
        assert_eq!(
            offset % self.alignment,
            0,
            "slice with offset {offset} needs to be a multiple of {}",
            self.alignment
        );
        let handle = SliceHandle::new();

        let storage = StorageHandle {
            id: storage_id,
            utilization: StorageUtilization { offset, size },
        };

        let padding = calculate_padding(size, self.alignment);

        Slice::new(storage, handle, padding)
    }

    /// Creates a page of given size by allocating on the storage.
    fn create_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<StorageId, IoError> {
        let storage = storage.alloc(self.page_size)?;

        let id = storage.id;
        self.ring.push_page(id);

        self.pages.insert(id, MemoryPage::new(HashMap::new()));

        Ok(id)
    }
}

impl Slice {
    pub(crate) fn split(&mut self, offset_slice: u64, buffer_alignment: u64) -> Option<Self> {
        let size_new = self.effective_size() - offset_slice;
        let offset_new = self.storage.offset() + offset_slice;
        let old_size = self.effective_size();

        let storage_new = StorageHandle {
            id: self.storage.id,
            utilization: StorageUtilization {
                offset: offset_new,
                size: size_new,
            },
        };

        self.storage.utilization = StorageUtilization {
            offset: self.storage.offset(),
            size: offset_slice,
        };

        if !offset_new.is_multiple_of(buffer_alignment) {
            panic!("slice with offset {offset_new} needs to be a multiple of {buffer_alignment}");
        }
        let handle = SliceHandle::new();
        if size_new < buffer_alignment {
            self.padding = old_size - offset_slice;
            assert_eq!(self.effective_size(), old_size);
            return None;
        }

        assert!(
            size_new >= buffer_alignment,
            "Size new > {buffer_alignment}"
        );
        self.padding = 0;
        let padding = calculate_padding(size_new - buffer_alignment, buffer_alignment);
        Some(Slice::new(storage_new, handle, padding))
    }

    pub(crate) fn next_slice_position(&self) -> u64 {
        self.storage.offset() + self.effective_size()
    }
}
