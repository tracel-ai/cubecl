use super::{calculate_padding, MemoryPool, Slice, SliceBinding, SliceHandle, SliceId};
use crate::{
    memory_management::{MemoryLock, MemoryUsage},
    storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization},
};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

/// A memory pool that allocates buffers in a range of sizes and reuses them to minimize allocations.
///
/// - Only one slice is supported per page, due to the limitations in WGPU where each buffer should only bound with
///   either read only or read_write slices but not a mix of both.
/// - The pool uses a ring buffer to efficiently manage and reuse pages.
pub struct ExclusiveMemoryPool {
    pages: HashMap<StorageId, MemoryPage>,
    slices: HashMap<SliceId, Slice>,
    ring_buffer: Vec<StorageId>,
    index: usize,
    max_page_size: u64,
    alignment: u64,
    dealloc_period: u64,
    last_dealloc: u64,
}

struct MemoryPage {
    slice_id: SliceId,
    dealloc_mark: bool,
}

impl ExclusiveMemoryPool {
    pub(crate) fn new(page_size: u64, alignment: u64, dealloc_period: u64) -> Self {
        // Pages should be allocated to be aligned.
        assert_eq!(page_size % alignment, 0);
        Self {
            pages: HashMap::new(),
            slices: HashMap::new(),
            ring_buffer: Vec::new(),
            index: 0,
            max_page_size: page_size,
            alignment,
            dealloc_period,
            last_dealloc: 0,
        }
    }

    /// Finds a free page that can contain the given size
    /// Returns a slice on that page if successful.
    fn get_free_page(&mut self, locked: Option<&MemoryLock>) -> Option<SliceId> {
        for _ in 0..self.ring_buffer.len() {
            let storage_id = &self.ring_buffer[self.index];
            if let Some(locked) = locked.as_ref() {
                if locked.is_locked(storage_id) {
                    continue;
                }
            }

            let page = self.pages.get_mut(storage_id).unwrap();
            page.dealloc_mark = false;
            let slice = self.slices.get(&page.slice_id).unwrap();
            self.index = (self.index + 1) % self.ring_buffer.len();
            if slice.handle.is_free() {
                return Some(page.slice_id);
            }
        }

        None
    }
}

impl MemoryPool for ExclusiveMemoryPool {
    /// Returns the resource from the storage, for the specified handle.
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices.get(binding.id()).map(|s| &s.storage)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        exclude: Option<&MemoryLock>,
    ) -> SliceHandle {
        let page = self.get_free_page(exclude);
        let slice_id = if let Some(page) = page {
            page
        } else {
            *self.alloc(storage, self.max_page_size).id()
        };

        let padding = calculate_padding(size, self.alignment);
        let slice = self.slices.get_mut(&slice_id).unwrap();
        // Return a smaller part of the slice. By construction, we only ever
        // get a page with a size > size, so this is ok to do.
        slice.storage.utilization = StorageUtilization { offset: 0, size };
        slice.padding = padding;
        slice.handle.clone()
    }

    fn alloc<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: u64) -> SliceHandle {
        let storage = storage.alloc(size);
        self.ring_buffer.push(storage.id);

        let handle = SliceHandle::new();
        let padding = calculate_padding(size, self.alignment);
        let slice = Slice::new(storage.clone(), handle, padding);

        let handle_slice = slice.handle.clone();
        let slice_id = *slice.handle.id();
        self.pages.insert(
            storage.id,
            MemoryPage {
                slice_id,
                dealloc_mark: false,
            },
        );
        self.slices.insert(slice_id, slice);
        handle_slice
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
            bytes_reserved: self.pages.len() as u64 * self.max_page_size,
        }
    }

    fn max_alloc_size(&self) -> u64 {
        self.max_page_size
    }

    fn cleanup<Storage: ComputeStorage>(&mut self, storage: &mut Storage, alloc_nr: u64) {
        let elapsed = alloc_nr - self.last_dealloc;

        if elapsed < self.dealloc_period {
            return;
        }

        self.last_dealloc = alloc_nr;

        let deallocations: HashSet<_> = self
            .pages
            .iter_mut()
            .filter_map(|(storage_id, page)| {
                let slice = self.slices.get(&page.slice_id).unwrap();

                if slice.is_free() {
                    // If not marked yet the memory might just have been freed.
                    if !page.dealloc_mark {
                        page.dealloc_mark = true;
                        None
                    } else {
                        Some(*storage_id)
                    }
                } else {
                    None
                }
            })
            .collect();

        // Perform any deallocations if necessary.
        if !deallocations.is_empty() {
            for storage_id in deallocations.iter() {
                let slice_id = self.pages[storage_id].slice_id;
                self.pages.remove(storage_id);
                self.slices.remove(&slice_id);
                storage.dealloc(*storage_id);
            }

            self.index = 0;
            self.ring_buffer
                .retain(|storage| !deallocations.contains(storage));
        }
    }
}
