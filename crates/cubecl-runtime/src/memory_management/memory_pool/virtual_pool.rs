use super::index::SearchIndex;
use super::{MemoryChunk, RingBuffer, SliceBinding, SliceHandle, SliceId};
use crate::memory_management::MemoryUsage;
use crate::memory_management::memory_pool::Slice;
use crate::memory_management::memory_pool::base::MemoryFragment;
use crate::storage::PhysicalStorageHandle;
use crate::storage::PhysicalStorageId;
use crate::storage::{StorageHandle, StorageId, StorageUtilization, VirtualStorage};
use crate::{memory_management::memory_pool::calculate_padding, server::IoError};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

const DEFRAG_THRESHOLD: usize = 20;
// Hardcoded the value for testing

/// A virtual slice that represents a fragment of a virtual memory page
/// backed by a physical handle (or not, if unmapped).
#[derive(Debug)]
pub(crate) struct VirtualSlice {
    /// The underlying slice information
    pub slice: Slice,
    /// Physical memory handle backing this slice
    pub physical_handle: Option<PhysicalStorageHandle>,
}

/// A Virtual Slice is of course a memory fragment.
impl MemoryFragment for VirtualSlice {
    type Key = SliceId;

    fn is_free(&self) -> bool {
        self.slice.is_free()
    }

    fn effective_size(&self) -> u64 {
        self.slice.effective_size()
    }

    fn id(&self) -> Self::Key {
        self.slice.id()
    }

    /// For a slice to be splitted the memory pool needs to unmap it first, taking the storage handle from it.
    /// Then it can map it back.
    /// Additionally, non-free virtual slices cannot be split as they might be currently in use by any kernel.
    fn split(&mut self, offset_slice: u64, buffer_alignment: u64) -> Option<Self> {
        if !self.is_free() || self.physical_handle.is_some() {
            return None;
        }

        if let Some(new_slice) = self.slice.split(offset_slice, buffer_alignment) {
            return Some(Self {
                slice: new_slice,
                physical_handle: None,
            });
        }
        None
    }

    fn next_slice_position(&self) -> u64 {
        self.slice.next_slice_position()
    }
}

impl VirtualSlice {
    pub fn new(slice: Slice, physical_handle: Option<PhysicalStorageHandle>) -> Self {
        Self {
            slice,
            physical_handle,
        }
    }

    pub fn storage_id(&self) -> StorageId {
        self.slice.storage.id
    }

    pub fn storage_offset(&self) -> u64 {
        self.slice.storage.offset()
    }

    pub fn storage_size(&self) -> u64 {
        self.slice.storage.size()
    }

    pub fn unmap(&mut self) -> Option<PhysicalStorageHandle> {
        self.physical_handle.take()
    }

    pub fn map(&mut self, handle: PhysicalStorageHandle) {
        self.physical_handle = Some(handle);
    }
}

/// A virtual memory page that can contain multiple slices
/// At the storage level,
#[derive(Debug)]
pub(crate) struct VirtualMemoryPage {
    /// Map from offset to slice ID
    pub slices: HashMap<u64, SliceId>,
    /// The virtual storage ID for this page
    pub storage_id: StorageId,
    /// Total size of this page
    pub size: u64,
    /// Id of the next memory page (linked list)
    pub next_page_id: Option<StorageId>
}

impl VirtualMemoryPage {
    pub fn new(storage_id: StorageId, size: u64) -> Self {
        Self {
            slices: HashMap::new(),
            storage_id,
            size,
            next_page_id: None
        }
    }

    /// Utilities to check if this memory page is completely free.
    pub fn is_completely_free(&self, slices: &HashMap<SliceId, VirtualSlice>) -> bool {
        self.slices
            .values()
            .all(|slice_id| slices.get(slice_id).is_none_or(|slice| slice.is_free()))
    }

    /// Utility to check if this memory page is empty
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Next page id:
    pub fn next_page(&self) -> Option<&StorageId> {
        self.next_page_id.as_ref()
    }
}

/// A Virtual memory page is also a memory chunk.
/// As stated in `base.rs`, the idea is that memory chunks represent contiguous memory regions.
/// Of course, virtual memory address spaces are always contiguous so this is a perfect fit here.
/// Need to re-think the names for these traits.
impl MemoryChunk for VirtualMemoryPage {
    type Fragment = VirtualSlice;
    type Key = SliceId;

    /// This method is no different from that of the memory page.
    /// It attempts to merge a slice at a given address on this page with the following one.
    /// It is maintained because it does not require API calls, which makes it more efficient.
    /// than unmapping and remapping to get a contiguous chunk.
    /// The advantage with virtual memory is that you can now merge memory chunks into a bigger, contiguous chunk,
    /// which could not be done on the sliced pool, that relied solely on physical memory.
    fn merge_with_next_slice(
        &mut self,
        first_slice_address: u64,
        slices: &mut HashMap<Self::Key, Self::Fragment>,
    ) -> bool {
        if self.is_empty() {
            return false;
        };

        let first_slice_id = match self.find_slice(first_slice_address) {
            Some(id) => id,
            None => return false,
        };

        let next_slice_address = {
            let first_slice = match slices.get(&first_slice_id) {
                Some(s) => s,
                None => return false,
            };
            first_slice_address + first_slice.effective_size()
        };

        if let Some(next_slice_id) = self.find_slice(next_slice_address) {
            let (next_slice_eff_size, next_slice_is_free) = {
                let next_slice = match slices.get(&next_slice_id) {
                    Some(s) => s,
                    None => return false,
                };
                (next_slice.effective_size(), next_slice.is_free())
            };

            if next_slice_is_free {
                let first_slice = slices.get_mut(&first_slice_id).unwrap();
                let first_slice_eff_size = first_slice.effective_size();
                let first_slice_offset = first_slice.slice.storage.offset();
                let merged_size = first_slice_eff_size + next_slice_eff_size;

                first_slice.slice.storage.utilization = StorageUtilization {
                    size: merged_size,
                    offset: first_slice_offset,
                };
                first_slice.slice.padding = 0;

                // Cleanup the extra slice
                self.slices.remove(&next_slice_address);
                slices.remove(&next_slice_id);
                return true;
            }
        }
        false
    }

    /// Find a slice inside this chunk.
    fn find_slice(&self, address: u64) -> Option<Self::Key> {
        self.slices.get(&address).copied()
    }

    /// Insert a slice at the specific address on this chunk.
    fn insert_slice(&mut self, address: u64, slice: Self::Key) {
        self.slices.insert(address, slice);
    }
}

/// A memory pool that uses virtual memory for efficient allocation and defragmentation
pub(crate) struct VirtualMemoryPool {
    /// Virtual memory pages
    pages: HashMap<StorageId, VirtualMemoryPage>,
    /// All virtual slices in the pool
    virtual_slices: HashMap<SliceId, VirtualSlice>,
    /// Index for efficient searching of physical handles by size
    physical_index: SearchIndex<PhysicalStorageId>,
    /// Free physical handles that can be reused
    free_physical_handles: HashMap<PhysicalStorageId, PhysicalStorageHandle>,
    /// Ring buffer for page management
    ring: RingBuffer,
    /// Storage index for pages
    storage_index: SearchIndex<StorageId>,
    /// Recently added pages
    /// Not sure about the purpose of this. Just saw it on the SlicedPool and thought it would be a good idea to keep track of recently added pages, then upon each defragmentation, we can avoid unmapping slices that belong to this set.
    recently_added_pages: HashSet<StorageId>,
    /// Recently allocated size
    recently_allocated_size: u64,
    /// Minimum allocation size
    min_alloc_size: u64,
    /// Maximum allocation size
    max_alloc_size: u64,
    /// Memory alignment
    alignment: u64,
    /// Page size
    page_size: u64,
    /// Keep track of last allocated page for better alignment:
    last_allocated_page: Option<StorageId>

}

impl VirtualMemoryPool {
    /// Virtual memory works best when the variability of the physical allocations is low.
    /// Therefore, provided that we can only allocate chunks that are multiples of the gpu page size (2MiB) on CUDA,
    /// Therefore, it might be useful to have buckets of free physical chunks given their size. I am using the search index here to search for physical allocations given their size for this purpose.
    pub fn new(min_alloc_size: u64, max_alloc_size: u64, page_size: u64, alignment: u64) -> Self {
        // Ensure the allocation range is aligned to the granularity.
        let min_alloc_size = min_alloc_size.saturating_sub(1).next_multiple_of(alignment);
        let max_alloc_size = max_alloc_size.saturating_sub(1).next_multiple_of(alignment);
        assert_eq!(
            page_size % alignment,
            0,
            "Page size must be aligned to {}.",
            alignment
        );

        Self {
            pages: HashMap::new(),
            virtual_slices: HashMap::new(),
            physical_index: SearchIndex::new(),
            free_physical_handles: HashMap::new(),
            ring: RingBuffer::new(alignment),
            storage_index: SearchIndex::new(),
            recently_added_pages: HashSet::new(),
            recently_allocated_size: 0,
            min_alloc_size,
            max_alloc_size,
            alignment,
            page_size,
            last_allocated_page: None
        }
    }

    // Find or allocate a physical handle of the required size
    fn get_or_alloc_physical<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<PhysicalStorageHandle, IoError> {
        let aligned_size = size.next_multiple_of(self.alignment);
        assert!(
            aligned_size > self.min_alloc_size && aligned_size < self.max_alloc_size,
            "Invalid allocation size. The minimum and maximum values for physical allocations are: {} and {}.",
            self.min_alloc_size,
            self.max_alloc_size
        );

        // Use SearchIndex to efficiently find a handle that's large enough
        // Range from aligned_size to [`max_alloc_size`] to get all handles >= aligned_size
        if let Some(id) = self
            .physical_index
            .find_by_size(aligned_size..self.max_alloc_size)
            .next()
            .cloned()
        {
            // Found a suitable handle, remove it from free list
            if let Some(handle) = self.free_physical_handles.remove(&id) {
                return Ok(handle);
            }
        }

        // No suitable handle found, allocate a new one
        let handle = storage.allocate(aligned_size)?;
        self.physical_index.insert(handle.id(), handle.size());
        Ok(handle)
    }

    /// Release a physical handle back to the free list
    fn release_physical(&mut self, handle: PhysicalStorageHandle) {
        let id = handle.id();
        self.free_physical_handles.insert(id, handle);
    }

    /// Create a new virtual page of the specified page size.
    fn create_virtual_page<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        previous_page: Option<StorageId>,
    ) -> Result<StorageId, IoError> {
        // Reserve virtual address space
        let handle = storage.reserve(self.page_size, previous_page)?;
        let storage_id = handle.id;

        // Create the page
        let page = VirtualMemoryPage::new(storage_id, self.page_size);
        self.pages.insert(storage_id, page);

        self.ring.push_page(storage_id);
        // I maintain a ring buffer and a storage index for pages to search them by page size.
        self.storage_index.insert(storage_id, self.page_size);

        Ok(storage_id)
    }

    /// Create a slice on a page
    fn create_slice(&self, offset: u64, size: u64, storage_id: StorageId) -> Slice {
        assert_eq!(
            offset % self.alignment,
            0,
            "Slice offset must be aligned to {}",
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

    /// Compact a page by merging adjacent free slices
    fn compact_page(&mut self, page_id: &StorageId) {
        if let Some(page) = self.pages.get_mut(page_id) {
            let page_addresses: Vec<u64> = page.slices.keys().cloned().collect();

            for address in page_addresses.iter() {
                let slice_id = page.find_slice(*address).unwrap();
                if let Some(slice) = self.virtual_slices.get(&slice_id)
                    && slice.is_free()
                    && page.merge_with_next_slice(*address, &mut self.virtual_slices)
                {
                    break;
                }
            }
        }
    }

    /// This method implements the main defragmentation algorithm of the memory spaces.
    /// Compacts pages, collects free slices and unmaps them from their physical handles.
    /// The handles then become available for use when [`try_reserve`] or [`alloc`] come in.
    pub fn defragment<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
    ) {
        // First compact all pages. This was already done by the [SlicedPool]
        let keys: Vec<StorageId> = self.pages.keys().cloned().collect();

        for id in keys {
            self.compact_page(&id);
        }
        // Collect all free slices.
        let free_slice_ids: Vec<SliceId> = self
            .virtual_slices
            .iter()
            .filter(|(_, vs)| vs.is_free() && !self.recently_added_pages.contains(&vs.storage_id()))
            .map(|(id, _)| *id)
            .collect();

        if free_slice_ids.is_empty() {
            return;
        }

        // Unmap all free slices, releasing the physical handles for reuse.
        // The physical handles will be automatically reused on any [`alloc`] or [`try_reserve`]
        // calls.
        for slice_id in &free_slice_ids {

            if let Some(mut slice) = self.virtual_slices.remove(slice_id) {
                // Check if we can unmap the slice and release the handle if so.
                if let Some(mut handle) = slice.unmap() {
                    storage.unmap(slice.storage_id(), slice.storage_offset(), &mut handle);
                    // Will be reused on alloc or reserve.
                    self.release_physical(handle);
                }
            }
        }

    }
}

impl VirtualMemoryPool {
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.virtual_slices
            .get(binding.id())
            .map(|vs| &vs.slice.storage)
    }

    /// Attempt to reserve an slice of an specific size
    fn try_reserve<Storage: VirtualStorage>(
        &mut self,
        size: u64,
        storage: &mut Storage,
    ) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        // Try to find a free slice using the ring buffer
        let slice_id =
            self.ring
                .find_free_slice(effective_size, &mut self.pages, &mut self.virtual_slices)?; // Return None if no slice is found.

        let old_slice = self.virtual_slices.get(&slice_id)?;
        let old_slice_offset = old_slice.slice.storage.offset();
        let id = old_slice.slice.storage.id;

        // Get or allocate physical handle if needed
        let physical_handle = if old_slice.physical_handle.is_none() {
            // Physical handles are released here after n allocations.
            let mut physical = self.get_or_alloc_physical(storage, effective_size).ok()?;
            storage.map(id, old_slice_offset, &mut physical).ok()?;

            Some(physical)
        } else {
            old_slice.physical_handle.clone()
        }?;

        let virtual_slice = self.virtual_slices.get_mut(&slice_id)?;
        virtual_slice.slice.padding = padding;
        virtual_slice.slice.storage.utilization.size = size;
        virtual_slice.map(physical_handle);

        Some(virtual_slice.slice.handle.clone())
    }

    /// Directly allocate a new slice using the backing storage.
    fn alloc<Storage: VirtualStorage>(
        &mut self,
        size: u64,
        alloc_counter: usize,
        storage: &mut Storage,
    ) -> Result<SliceHandle, IoError> {


        /// Defragment and reset tracking every DEFRAG_THRESHOLD allocations.
        if alloc_counter % DEFRAG_THRESHOLD == 0 {
            self.defragment(storage);
            self.recently_added_pages.clear();
            self.recently_allocated_size = 0;
        };

        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;


        // Create a new virtual page
        let page_id = self.create_virtual_page(storage, self.last_allocated_page)?; // Attempt to allocate the page contiguous to the last one always. This should enforce better page alignment.


        // Maintain the chain of contiguous pages.
        if let Some(last_allocation) = self.last_allocated_page {
            if let Some(last_page) = self.pages.get_mut(&last_allocation) {
                last_page.next_page_id = Some(page_id);
            };
        };

        self.last_allocated_page = Some(page_id);
        self.recently_added_pages.insert(page_id);
        self.recently_allocated_size += self.page_size;

        // Create the main slice
        let slice = self.create_slice(0, size, page_id);
        let slice_handle = slice.handle.clone();
        let slice_id = slice.id();
        let slice_offset = slice.storage.offset();

        // Get physical backing
        let mut physical = self.get_or_alloc_physical(storage, effective_size)?;

        // Map physical to virtual
        storage.map(page_id, slice_offset, &mut physical)?;

        // Create virtual slice
        let virtual_slice = VirtualSlice::new(slice, Some(physical));
        self.virtual_slices.insert(slice_id, virtual_slice);

        // Update page
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.insert_slice(slice_offset, slice_id);
        }

        // Create extra slice if needed
        if effective_size < self.page_size {
            let extra_slice =
                self.create_slice(effective_size, self.page_size - effective_size, page_id);
            let extra_id = extra_slice.id();
            let extra_offset = extra_slice.storage.offset();
            let extra_virtual = VirtualSlice::new(extra_slice, None);
            self.virtual_slices.insert(extra_id, extra_virtual);

            if let Some(page) = self.pages.get_mut(&page_id) {
                page.insert_slice(extra_offset, extra_id);
            }
        }

        Ok(slice_handle)
    }

    /// Collect memory usage stats.
    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .virtual_slices
            .values()
            .filter(|vs| !vs.is_free())
            .collect();

        let total_page_size: u64 = self.pages.values().map(|p| p.size).sum();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|vs| vs.slice.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|vs| vs.slice.padding).sum(),
            bytes_reserved: total_page_size,
        }
    }

    /// Clean up all physical memory handles that have become free after defragmentation.
    fn cleanup<Storage: VirtualStorage>(
        &mut self,
        _alloc_nr: u64,
        explicit: bool,
        storage: &mut Storage,
    ) {
        if explicit {
            self.defragment(storage);

            for (id, _handle) in self.free_physical_handles.drain() {
                self.physical_index.remove(&id);
                storage.release(id)
            }
            // Not sure if it is a good idea to release virtual memory from pages that have become full free.
        }
    }
}
