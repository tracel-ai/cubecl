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
    /// Physical memory handles backing this slice
    pub physical_handles: Option<Vec<PhysicalStorageHandle>>,
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

    /// Non-free virtual slices cannot be split as they might be currently in use by any kernel.
    fn split(&mut self, offset_slice: u64, buffer_alignment: u64) -> Option<Self> {
        if !self.is_free() {
            return None;
        }

        let size_new = self.effective_size() - offset_slice;

        if let Some(new_slice) = self.slice.split(offset_slice, buffer_alignment) {
            // Split the physical handles of this slice.
            let physical_handles = if let Some(handles) = self.physical_handles.as_mut() {
                let mut total_size = 0;
                let mut new_handles: Vec<PhysicalStorageHandle> = Vec::new();

                while let Some(handle) = handles.pop() {
                    total_size += handle.size();

                    if total_size >= size_new {
                        break;
                    }

                    new_handles.push(handle);
                }

                Some(new_handles)
            } else {
                None
            };

            return Some(Self {
                slice: new_slice,
                physical_handles,
            });
        }
        None
    }

    fn next_slice_position(&self) -> u64 {
        self.slice.next_slice_position()
    }
}

impl VirtualSlice {
    pub fn new(slice: Slice, physical_handles: Option<Vec<PhysicalStorageHandle>>) -> Self {
        Self {
            slice,
            physical_handles,
        }
    }

    pub fn storage_id(&self) -> StorageId {
        self.slice.storage.id
    }

    pub fn storage_offset(&self) -> u64 {
        self.slice.storage.offset()
    }

    pub fn set_offset(&mut self, offset: u64) {
        self.slice.storage.utilization.offset = offset;
    }

    pub fn set_page(&mut self, page: StorageId) {
        self.slice.storage.id = page;
    }

    pub fn storage_size(&self) -> u64 {
        self.slice.storage.size()
    }

    pub fn unmap(&mut self) -> Option<Vec<PhysicalStorageHandle>> {
        self.physical_handles.take()
    }

    pub fn map(&mut self, handles: Vec<PhysicalStorageHandle>) {
        self.physical_handles = Some(handles);
    }

    fn is_mapped(&self) -> bool {
        self.physical_handles.is_some()
    }


    /// Check the total size of this handle that is mapped
    fn mapped_size(&self) -> u64 {
        if let Some(handles) = &self.physical_handles {
            return handles.iter().map(|h| h.size()).sum();
        }
        0u64
    }



    pub fn set_size(&mut self, size: u64) {
        self.slice.storage.utilization.size = size;
    }

    pub fn set_padding(&mut self, padding: u64) {
        self.slice.padding = padding;
    }
}

/// A virtual memory page that can contain multiple slices
/// At the storage level,
#[derive(Debug)]
pub(crate) struct VirtualMemoryPage {
    /// Map from offset to slice ID
    /// Uses a btree map instead of hashmap to ensure offsets are ordered.
    pub slices: HashMap<u64, SliceId>,
    /// The virtual storage ID for this page
    pub storage_id: StorageId,
    /// Total size of this page
    pub size: u64,
    /// Id of the next memory page (linked list)
    pub next_page_id: Option<StorageId>,
}

impl VirtualMemoryPage {
    pub fn new(storage_id: StorageId, size: u64) -> Self {
        Self {
            slices: HashMap::new(),
            storage_id,
            size,
            next_page_id: None,
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


        // Return the second slice, but also check if the first slice is mapped.
        let next_slice_address = {
            let first_slice = match slices.get(&first_slice_id) {
                Some(s) => s,
                None => return false,
            };
            first_slice_address + first_slice.effective_size()
        };

        // Search for the second slice.
        if let Some(next_slice_id) = self.find_slice(next_slice_address) {
            let (next_slice_eff_size, next_slice_is_free, next_slice_handles) = {
                let next_slice = match slices.get(&next_slice_id) {
                    Some(s) => s,
                    None => return false,
                };

                let next_slice_data = if next_slice.is_free() && next_slice.is_mapped() {
                    next_slice.physical_handles.clone()
                } else {
                    None
                };

                (
                    next_slice.effective_size(),
                    next_slice.is_free(),
                    next_slice_data,
                )
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

                // We need to merge the physical handles here in case they exist.
                if next_slice_handles.is_some()
                    && let Some(handles) = first_slice.physical_handles.as_mut()
                {
                    handles.extend(next_slice_handles.unwrap());

                } else if next_slice_handles.is_some() {

                    // Note that a completely valid slice needs to be either completely mapped or unmapped
                    // If you are carefully reading the code you will notice that here we might be combining a non-mapped slice with a mapped one, which could result in a partial mapping and would not be valid.
                    // To account for that, we fill the missing mappings in the [`try_reserve`] method.
                    // This is an optimization I have added in order to reduce the total number of API calls to map memory.
                    first_slice.map(next_slice_handles.unwrap());
                }
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
    /// Ensures we cannot insert two slices at the same offset in the page.
    fn insert_slice(&mut self, address: u64, slice: Self::Key) {
        if self.slices.get(&address).is_none() {
            self.slices.insert(address, slice);
        };
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
    /// This flags allow to have a fully contiguous virtual address space. The allocation workflow is the following:
    /// |------------------|    |------------------|            |------------------|
    /// |  page 1          | -> |  page 2          | [.....] -> |  page n          |
    /// |------------------|    |------------------|            |------------------|
    /// Where page n - 1 starts where page n ends and so on.
    /// Pages are memory ranges that are contiguous by definition, and slices inside pages can be merged and split on demand, without the need of any API calls.
    /// To merge a range of pages into a single one we will need an API call to the driver, although if pages are actually contiguous it is not necessary.
    /// Keep track of last allocated page for better alignment:
    last_allocated_page: Option<StorageId>,
    /// Keep track of the first page that was allocated after last defragmentation
    first_page: Option<StorageId>,
}

impl VirtualMemoryPool {
    /// Virtual memory works best when the variability of the physical allocations is low.
    /// Therefore, provided that we can only allocate chunks that are multiples of the gpu page size (2MiB) on CUDA,
    /// Therefore, it might be useful to have buckets of free physical chunks given their size. I am using the search index here to search for physical allocations given their size for this purpose.
    pub fn new(min_alloc_size: u64, max_alloc_size: u64, alignment: u64) -> Self {
        // Ensure the allocation range is aligned to the granularity.
        let min_alloc_size = min_alloc_size.saturating_sub(1).next_multiple_of(alignment);
        let max_alloc_size = max_alloc_size.saturating_sub(1).next_multiple_of(alignment);


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
            last_allocated_page: None,
            first_page: None,
        }
    }

    // Find or allocate a physical handle of the required size
    fn get_or_alloc_physical<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<PhysicalStorageHandle, IoError> {

        // Physical memory should always be aligned.
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
        page_size: u64,
        previous_page: Option<StorageId>,
    ) -> Result<StorageId, IoError> {
        // Reserve virtual address space of the required size.
        // An advantage of using this memory pool over the sliced pool is that virtual pages are dynamically sized.
        // This allows to make certain optimizations, like merging all free pages into a single big virtual page.
        let handle = storage.reserve(page_size, previous_page)?;
        let storage_id = handle.id;

        // Create the page
        let page = VirtualMemoryPage::new(storage_id, page_size);
        self.pages.insert(storage_id, page);

        self.ring.push_page(storage_id);
        // I maintain a ring buffer and a storage index for pages to search them by page size.
        self.storage_index.insert(storage_id, page_size);

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

    /// Completely compact a page by merging adjacent free slices
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
    pub fn defragment<Storage: VirtualStorage>(&mut self, storage: &mut Storage) {
        // First merge all pages
        if let Some(first) = self.first_page {
            let mut current = first;

            while let Some(next) = self.pages.get(&current).unwrap().next_page().cloned() {
                // Query the storage to check if they are aligned.
                if storage.are_aligned(&current, &next) {
                    // Get the size of the first page to calculate the offset
                    let page_1_size = self.pages.get(&current).unwrap().size;

                    // Remove the second page to avoid borrow checker issues
                    let mut page_2 = self.pages.remove(&next).unwrap();

                    // Move all slices from page_2 to page_1 with adjusted offsets
                    let slices_to_move: Vec<(u64, SliceId)> = page_2.slices.drain().collect();

                    for (old_offset, slice_id) in slices_to_move {
                        let new_offset = page_1_size + old_offset;

                        // Update the slice's storage information
                        if let Some(virtual_slice) = self.virtual_slices.get_mut(&slice_id) {
                            virtual_slice.slice.storage.id = current; // Change to first page's ID
                            virtual_slice.slice.storage.utilization.offset = new_offset;
                        }

                        // Insert the slice into the first page with new offset
                        self.pages
                            .get_mut(&current)
                            .unwrap()
                            .slices
                            .insert(new_offset, slice_id);
                    }

                    // Update the first page's size to include the second page
                    self.pages.get_mut(&current).unwrap().size += page_2.size;

                    // Update the next page pointer to skip the merged page
                    self.pages.get_mut(&current).unwrap().next_page_id = page_2.next_page_id;

                    // Remove the second page from ring buffer and storage index
                    self.storage_index.remove(&next);
                    self.ring.remove_page(&next);

                    // Continue with the same current page since we merged the next one
                    continue;
                }

                current = next;
            }
        }

        // At this point, pages have been merged into a single one big chunk.
        // Therefore we can now try to compact them, merging all contiguous free slices.
        // In the sliced pool, you could only compact at a single page level, therefore the maximum compactability you could achieve if of size [`page_size`]

        let ids: Vec<StorageId> = self.pages.keys().cloned().collect();
        for id in ids {
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
                if let Some(handles) = slice.unmap() {


                    for mut handle in handles {
                        storage.unmap(slice.storage_id(), slice.storage_offset(), &mut handle);
                        // Will be reused on alloc or reserve.
                        self.release_physical(handle);
                    }
                }
            }
        }
    }
}

impl VirtualMemoryPool {
    pub fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    pub fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.virtual_slices
            .get(binding.id())
            .map(|vs| &vs.slice.storage)
    }

    /// Attempt to reserve an slice of an specific size
    /// Will search for the first free slice that has enough size and split it if necessary.
    /// If the slice is not mapped, it will map it automatically.
    pub fn try_reserve<Storage: VirtualStorage>(
        &mut self,
        size: u64,
        storage: &mut Storage,
    ) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        // Try to find a free slice using the ring buffer
        // This method will automatically find the first slice that is bigger than [`effective_size`], and split it if necessary.
        let slice_id =
            self.ring
                .find_free_slice(effective_size, &mut self.pages, &mut self.virtual_slices)?; // Return None if no slice is found.

        let old_slice = self.virtual_slices.get(&slice_id)?;
        let old_slice_offset = old_slice.storage_offset();
        let id = old_slice.storage_id();

        // Get or allocate physical handle if needed
        // Note that as free slices from very old pages are unmapped during defragmentation to reuse their handles,
        // we might see here that the slice is not mapped, so just in case we map it if necessary to ensure memory is always valid.
        // CASE A: COMPLETELY UNMAPPED SLICE
        let physical_handles = if !old_slice.is_mapped() {
            // Physical handles are released here after n allocations.
            let mut physical = self.get_or_alloc_physical(storage, effective_size).ok()?;
            storage.map(id, old_slice_offset, &mut physical).ok()?;
            Some(vec![physical])

        // CASE B: PARTIALLY MAPPED SLICE
        } else if old_slice.mapped_size() < effective_size {
            let mut handles = old_slice.physical_handles.clone().unwrap();
            let size_to_map = effective_size - old_slice.mapped_size();
            let mut physical = self.get_or_alloc_physical(storage, size_to_map).ok()?;
            storage.map(id, old_slice_offset, &mut physical).ok()?;
            handles.push(physical);
            Some(handles)

        }// BEST CASE: COMPLETELY MAPPED:
        else {
            old_slice.physical_handles.clone()
        };

        let virtual_slice = self.virtual_slices.get_mut(&slice_id)?;
        virtual_slice.set_padding(padding);
        virtual_slice.set_size(size);

        if let Some(ph_handles) = physical_handles {

            virtual_slice.map(ph_handles);

        };

        Some(virtual_slice.slice.handle.clone())
    }

    /// Directly allocate a new slice using the backing storage.
    pub fn alloc<Storage: VirtualStorage>(
        &mut self,
        size: u64,
        alloc_counter: usize,
        storage: &mut Storage,
    ) -> Result<SliceHandle, IoError> {
        // Defragment and reset tracking every DEFRAG_THRESHOLD allocations.
        if alloc_counter % DEFRAG_THRESHOLD == 0 {
            self.defragment(storage);
            self.recently_added_pages.clear();
            self.recently_allocated_size = 0;
            self.first_page = None;
        };

        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;

        // Create a new virtual page.
        // The idea is that first page reservation always is of effective_size.
        // This minimizes extra padding and enforces alignment, as we here do not need fixed-size pages.
        let page_id =
            self.create_virtual_page(storage, effective_size, self.last_allocated_page)?;
        // Attempt to allocate the page contiguous to the last allocated one always. This should enforce better page alignment.

        if self.first_page.is_none() {
            self.first_page = Some(page_id)
        };

        // Maintain the chain of contiguous pages.
        if let Some(last_allocation) = self.last_allocated_page {
            if let Some(last_page) = self.pages.get_mut(&last_allocation) {
                last_page.next_page_id = Some(page_id);
            };
        };

        // Update tracking structures.
        self.last_allocated_page = Some(page_id);
        self.recently_added_pages.insert(page_id);
        self.recently_allocated_size += effective_size;

        // Create the main slice on the page.
        let slice = self.create_slice(0, size, page_id);
        let slice_handle = slice.handle.clone();
        let slice_id = slice.id();
        let slice_offset = slice.storage.offset();

        // Get physical backing
        let mut physical = self.get_or_alloc_physical(storage, effective_size)?;
        // Map physical to virtual
        storage.map(page_id, slice_offset, &mut physical)?;
        // Create virtual slice
        let virtual_slice = VirtualSlice::new(slice, Some(vec![physical]));
        self.virtual_slices.insert(slice_id, virtual_slice);

        // Update page, inserting the slice at the correct offset.
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.insert_slice(slice_offset, slice_id);
        }
        // We do not need to create an extra slice, as the created slice currently is of the same size as the page.
        Ok(slice_handle)
    }

    /// Collect memory usage stats.
    pub fn get_memory_usage(&self) -> MemoryUsage {
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
    pub fn cleanup<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
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
