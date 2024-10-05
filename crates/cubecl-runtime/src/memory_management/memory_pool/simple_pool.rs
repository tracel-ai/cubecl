use super::{
    calculate_padding, MemoryPool, MemoryUsage, Slice, SliceBinding, SliceHandle, SliceId,
};
use crate::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// A memory pool that allocates buffers in a range of sizes and reuses them to minimize allocations.
///
/// - Only one slice is supported per page, due to the limitations in WGPU where each buffer should only bound with
///   either read only or read_write slices but not a mix of both.
/// - The pool uses a ring buffer to efficiently manage and reuse pages.
///
/// Fields:
/// - `pages`: A hashmap storing the allocated pages by their IDs.
/// - `slices`: A hashmap storing the slices by their IDs.
/// - `ring_buffer`: A vector used as a ring buffer to manage page reuse.
/// - `index`: The current position in the ring buffer.
pub struct SimpleMemoryPool {
    pages: HashMap<StorageId, MemoryPage>,
    slices: HashMap<SliceId, Slice>,
    ring_buffer: Vec<StorageId>,
    index: usize,
    max_page_size: usize,
    alignment: usize,
}

struct MemoryPage {
    slice_id: SliceId,
}

impl MemoryPool for SimpleMemoryPool {
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
        size: usize,
        exclude: &[StorageId],
    ) -> SliceHandle {
        let page = self.get_free_page(exclude);
        let slice_id = if let Some(page) = page {
            page.slice_id
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

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> SliceHandle {
        let storage = storage.alloc(size);
        self.ring_buffer.push(storage.id);

        let handle = SliceHandle::new();
        let padding = calculate_padding(size, self.alignment);
        let slice = Slice::new(storage.clone(), handle, padding);

        let handle_slice = slice.handle.clone();
        let slice_id = *slice.handle.id();
        self.pages.insert(storage.id, MemoryPage { slice_id });
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
            number_allocs: used_slices.len(),
            bytes_in_use: used_slices.iter().map(|s| s.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|s| s.padding).sum(),
            bytes_reserved: self.pages.len() * self.max_page_size,
        }
    }

    fn max_alloc_size(&self) -> usize {
        self.max_page_size
    }
}

impl SimpleMemoryPool {
    pub(crate) fn new(page_size: usize, alignment: usize) -> Self {
        // Pages should be allocated to be aligned.
        assert_eq!(page_size % alignment, 0);
        Self {
            pages: HashMap::new(),
            slices: HashMap::new(),
            ring_buffer: Vec::new(),
            index: 0,
            max_page_size: page_size,
            alignment,
        }
    }

    /// Finds a free page that can contain the given size
    /// Returns a slice on that page if sucessfull.
    fn get_free_page(&mut self, exclude: &[StorageId]) -> Option<&MemoryPage> {
        for _ in 0..self.ring_buffer.len() {
            let storage_id = &self.ring_buffer[self.index];
            if exclude.contains(storage_id) {
                continue;
            }

            let page = self.pages.get(storage_id).unwrap();
            let slice = self.slices.get(&page.slice_id).unwrap();
            self.index = (self.index + 1) % self.ring_buffer.len();
            if slice.handle.is_free() {
                return Some(page);
            }
        }

        None
    }
}
