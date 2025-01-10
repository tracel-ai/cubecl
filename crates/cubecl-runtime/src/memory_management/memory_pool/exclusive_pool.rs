use super::{calculate_padding, MemoryPool, Slice, SliceBinding, SliceHandle, SliceId};
use crate::{
    memory_management::MemoryUsage,
    storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization},
};
use alloc::vec::Vec;
use hashbrown::HashSet;

/// A memory pool that allocates buffers in a range of sizes and reuses them to minimize allocations.
///
/// - Only one slice is supported per page, due to the limitations in WGPU where each buffer should only bound with
///   either read only or read_write slices but not a mix of both.
/// - The pool uses a ring buffer to efficiently manage and reuse pages.
pub struct ExclusiveMemoryPool {
    pages: Vec<MemoryPage>,
    dealloc_marked: HashSet<StorageId>,
    page_size: u64,
    alignment: u64,
    dealloc_period: u64,
    last_dealloc: u64,
}

struct MemoryPage {
    slice_id: SliceId,
    slice: Slice,
}

impl ExclusiveMemoryPool {
    pub(crate) fn new(page_size: u64, alignment: u64, dealloc_period: u64) -> Self {
        // Pages should be allocated to be aligned.
        assert_eq!(page_size % alignment, 0);
        Self {
            pages: Vec::new(),
            dealloc_marked: HashSet::new(),
            page_size,
            alignment,
            dealloc_period,
            last_dealloc: 0,
        }
    }

    /// Finds a free page that can contain the given size
    /// Returns a slice on that page if successful.
    fn get_free_page(&mut self) -> Option<&mut MemoryPage> {
        // Start iteratng from first free page index.
        self.pages.iter_mut().find(|page| page.slice.is_free())
    }

    fn alloc_page<Storage: ComputeStorage>(&mut self, storage: &mut Storage) -> &mut MemoryPage {
        let storage = storage.alloc(self.page_size);

        let handle = SliceHandle::new();
        let padding = calculate_padding(self.page_size, self.alignment);
        let slice = Slice::new(storage.clone(), handle, padding);

        let slice_id = *slice.handle.id();
        self.pages.push(MemoryPage { slice, slice_id });
        let idx = self.pages.len() - 1;
        &mut self.pages[idx]
    }
}

impl MemoryPool for ExclusiveMemoryPool {
    /// Returns the resource from the storage, for the specified handle.
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        let binding_id = *binding.id();
        self.pages
            .iter()
            .find(|p| p.slice_id == binding_id)
            .map(|p| &p.slice.storage)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alignment);

        let page = self.get_free_page();

        if let Some(page) = page {
            // Return a smaller part of the slice. By construction, we only ever
            // get a page with a big enough size, so this is ok to do.
            page.slice.storage.utilization = StorageUtilization { offset: 0, size };
            page.slice.padding = padding;

            let id = page.slice.storage.id;
            let handle = page.slice.handle.clone();

            // If this is the base allocation, mark page as used.
            // Otherwise, let it go if noone else needs it.
            self.dealloc_marked.remove(&id);

            return Some(handle);
        }

        None
    }

    fn alloc<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: u64) -> SliceHandle {
        assert!(size <= self.page_size);
        let padding = calculate_padding(size, self.alignment);
        let page = self.alloc_page(storage);
        // Return a smaller part of the slice. By construction, we only ever
        // get a page with a big enough size, so this is ok to do.
        page.slice.storage.utilization = StorageUtilization { offset: 0, size };
        page.slice.padding = padding;
        page.slice.handle.clone()
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .pages
            .iter()
            .filter(|page| !page.slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|s| s.slice.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|s| s.slice.padding).sum(),
            bytes_reserved: self.pages.len() as u64 * self.page_size,
        }
    }

    fn max_alloc_size(&self) -> u64 {
        self.page_size
    }

    fn cleanup<Storage: ComputeStorage>(&mut self, storage: &mut Storage, alloc_nr: u64) {
        if alloc_nr - self.last_dealloc < self.dealloc_period {
            return;
        }

        self.last_dealloc = alloc_nr;

        self.pages.retain_mut(|page| {
            if page.slice.is_free() && !self.dealloc_marked.insert(page.slice.storage.id) {
                // Dealloc page.
                storage.dealloc(page.slice.storage.id);
                return false;
            }

            true
        });
    }
}
