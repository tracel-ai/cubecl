/// Virtual Memory Pool (still do not know how to name things in CS)
///
/// Here are some intuitions about my implementation:
/// There are three main concepts:
///
/// 1. `Physical handles.` They are blocks of physical memory located anywhere on the device that we can ask the `VirtualStorage` for. However, they are not usable.
///
/// 2. `Memory Pages`: following the implementation of the Sliced Pool, a memory page is some chunk of device memory that is guaranteed to be aligned, identified by a storage id. On the device side, they represent virtual address spaces that can be incrementally mapped on demand.
///
/// 3. `Virtual Slices`: These represent a fragment of a memory page that can be backed by physical handles (mapped) or not (unmapped). They are identified by specific offsets at the memory pages. This allows us to  compact pages by merging consecutive free slices. A slice is considered free in CubeCL if the shared reference counter of its handle is 0.
///
/// Note that when a user requests for memory, we give him (or her), an slice handle, which includes a pointer (storage id) to the main page where this handle belongs to and an offset within the handle.
///
/// It is the responsibility of the memory system to guarantee that the handle is valid.
///Therefore, before each allocation, we ensure all the physical handles backing the slice are mapped.
///
/// At allocation time, there will be only one physical handle backing the slice.
/// This phyical handle will be the same size as the slice size.
/// If the slice is merged with other slices in the same page, their physical handles will be combined.
/// When a slice reused, it is sometimes split in two. At this point it must be either unmapped (physical_handles is `None`) or cut up at some point in the physical handles vector moving part of the handles to the new slice, while maintining the order.
///
/// Therefore, this memory pool will work better if the physical handles are of a small, predictable size.
/// Both the parameter [`max_alloc_size`] and [`àlignment`] are left to be chosen by the user, but in practice this pools works best when the difference between them is small.
///
/// If your workload is going to require very big tensors, you could keep them the same value (p.eg  20 MiB which is what PyTorch uses). In cases where the size of tensors is expected to be smaller, you can increase the variability and limit alignment to be the device [`minimum granularity`] which should match gpu page size.
///
/// In both cases, having a ratio between alignment and max_alloc_size close to 1 will be in favor of better physical memory reuse.
///
/// Implementation explanations:
///
/// Consider the following:
///
/// We have a mapped slice (V1) with the following layout:
///
/// ------------------------------
/// | Ph1 | Ph2 | Ph3 | Ph5       | -> [V1] [ ADDR + SIZE + OFFSET]
/// ------------------------------
///
/// Where all handles are of size `alignment`(but ph5 which is size = alignment * 2)
///
/// Then, V1 can be part of a memory page (P0), which is just an address space that is contiguous to the user, with the following layout:
///
/// P0 -> VIRTUAL ADDR : 0             [V0] OFFSET = 0; SIZE = PH1 + PH2 + PH3 + PH4 + PH5
///    -> VIRTUAL ADDR: 0 + [Size(v0)] [V1] OFFSET = PH0; SIZE = PH1 + PH2 + PH3 + PH4 + PH5
///
/// Additionally, pages are allocated in a contiguous manner, leveraging the power of [`cuMemAddressReserve`] which allows you to provide a hint to the driver in order to allocate virtual memory at a specific address.
///
/// So then, the next page will be allocated:
///
/// P1 -> VIRTUAL ADDR : P0_SIZE    [V2] OFFSET = 0, SIZE = P6
///
/// At defragmentation (periodically every N allocations) the following will happen (assuming all alices are free):
///
/// 1. P0 and P1 will be combined in a single logical page (P0), without making any call to the driver or moving memory.
/// 2. The remaining page (P0) will be compacted by merging adjacent pairs of free slices into a single slice, which will now consist of multiple handles and will be ready to be split at next reservation.
///
/// Example:
///
/// When both slices V0, V1, V2 are free, we are allowed to merge them in a single bigger slice:
///
/// P0 -> VIRTUAL ADDR : 0             [V0] OFFSET = 0; SIZE = PH1 + PH2 + PH3 + PH4 + PH5 + P0 + V2-SIZE
///
/// Notice that V0 and V1 were mapped, but V2 was not (this is a pretty rare case, but I wanted to show it here in order to demonstrate the actual workflow). Therefore the new big slice is not completely mapped. This can cause problems if we then try to split the slice to get the unmapped portion.
///
/// To avoid that, at reservation time, we allocate a handle of enough size to fill-in the missing size of the partially-mapped slice. If the required size for the handle is bigger than max_alloc_size, we allocate 'N' handles until we match target.
/// As allocations should always be aligned to [`alignment`] (in practice the minimum granularity or GPU page size), the case where the required handle is smaller than `min_alloc_size` (which equals `alignment`) never happens.
use super::{MemoryChunk, RingBuffer, SliceBinding, SliceHandle, SliceId};
use crate::memory_management::MemoryUsage;
use crate::memory_management::memory_pool::MemoryPool;
use crate::memory_management::memory_pool::Slice;
use crate::memory_management::memory_pool::base::MemoryFragment;
use crate::storage::ComputeStorage;
use crate::storage::PhysicalStorageHandle;
use crate::storage::PhysicalStorageId;
use crate::storage::{StorageHandle, StorageId, StorageUtilization, VirtualStorage};
use crate::{memory_management::memory_pool::calculate_padding, server::IoError};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

/// A virtual slice that represents a fragment of a virtual memory page
/// backed by a physical handle (or not, if unmapped).
#[derive(Debug)]
pub(crate) struct VirtualSlice {
    /// The underlying slice information
    pub slice: Slice,
    /// Physical memory handles backing this slice.
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

                // Here we split the handles of the slice.
                // This is tricky since we want to ensure handle order is kept correct.
                while let Some(handle) = handles.pop() {
                    total_size += handle.size();

                    if total_size >= size_new {
                        break;
                    }

                    new_handles.push(handle);
                }
                // Use reverse here to maintain handles order.
                new_handles.reverse();
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
    /// Total size of this page
    pub size: u64,
    /// Id of the next memory page (linked list)
    pub next_page_id: Option<StorageId>,
}

impl VirtualMemoryPage {
    pub fn new(size: u64) -> Self {
        Self {
            slices: HashMap::new(),
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
                } else if let Some(item) = next_slice_handles {
                    // Note that a completely valid slice needs to be either completely mapped or unmapped
                    // If you are carefully reading the code you will notice that here we might be combining a non-mapped slice with a mapped one, which could result in a partial mapping and would not be valid.
                    // To account for that, we fill the missing mappings in the [`try_reserve`] method.
                    // This is an optimization I have added in order to reduce the total number of API calls to map memory.
                    first_slice.map(item);
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
    pub(crate) pages: HashMap<StorageId, VirtualMemoryPage>,
    /// All virtual slices in the pool
    pub(crate) virtual_slices: HashMap<SliceId, VirtualSlice>,
    /// Queue  for efficient popping of free physical handles
    pub(crate) free_physical_queue: VecDeque<PhysicalStorageId>,
    /// Free physical handles that can be reused
    pub(crate) physical_handles: HashMap<PhysicalStorageId, PhysicalStorageHandle>,
    /// Ring buffer for page management
    pub(crate) ring: RingBuffer,
    /// Recently added pages
    /// Not sure about the purpose of this. Just saw it on the SlicedPool and thought it would be a good idea to keep track of recently added pages, then upon each defragmentation, we can avoid unmapping slices that belong to this set.
    pub(crate) recently_added_pages: HashSet<StorageId>,
    /// Recently allocated size
    pub(crate) recently_allocated_size: u64,
    /// Physical allocation size
    pub(crate) alloc_size: u64,
    /// This flags allow to have a fully contiguous virtual address space. The allocation workflow is the following:
    /// |------------------|    |------------------|            |------------------|
    /// |  page 1          | -> |  page 2          | [.....] -> |  page n          |
    /// |------------------|    |------------------|            |------------------|
    /// Where page n - 1 starts where page n ends and so on.
    /// Pages are memory ranges that are contiguous by definition, and slices inside pages can be merged and split on demand, without the need of any API calls.
    /// To merge a range of pages into a single one we will need an API call to the driver, although if pages are actually contiguous it is not necessary.
    /// Keep track of last allocated page for better alignment:
    pub(crate) last_allocated_page: Option<StorageId>,
    /// Keep track of the first page that was allocated after last defragmentation
    pub(crate) first_page: Option<StorageId>,
    // Dealloc or defragmentation period
    pub(crate) dealloc_period: u64,
}

impl VirtualMemoryPool {
    /// Virtual memory works best when the variability of the physical allocations is low.
    /// Therefore, provided that we can only allocate chunks that are multiples of the gpu page size (2MiB) on CUDA, it might be useful to have buckets of free physical chunks given their size. I am using the search index here to search for physical allocations given their size for this purpose.
    pub fn new(alloc_size: u64, min_granularity: u64, dealloc_period: u64) -> Self {
        // Ensure the allocation size is aligned.
        let alloc_size = alloc_size.next_multiple_of(min_granularity);
        // As all allocations need to be the same size, if alloc_size is bigger than alignment we need to align to `alloc_size`

        Self {
            pages: HashMap::new(),
            virtual_slices: HashMap::new(),
            free_physical_queue: VecDeque::new(),
            physical_handles: HashMap::new(),
            ring: RingBuffer::new(alloc_size),
            recently_added_pages: HashSet::new(),
            recently_allocated_size: 0,
            alloc_size,
            last_allocated_page: None,
            first_page: None,
            dealloc_period,
        }
    }

    // Find or allocate a physical handle of the required size
    fn get_or_alloc_physical<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<Vec<PhysicalStorageHandle>, IoError> {
        // Physical memory should always be aligned.
        let aligned_size = size.next_multiple_of(self.alloc_size);
        let num_blocks = aligned_size.div_ceil(self.alloc_size);

        let mut allocated_size = 0;
        let mut physical_handles = Vec::with_capacity(num_blocks as usize);

        // Pop from the queue.
        while let Some(id) = self.free_physical_queue.pop_front() {
            // Found a suitable handle, get it from the hashmap
            if let Some(handle) = self.physical_handles.get(&id) {
                physical_handles.push(handle.clone());
                allocated_size += self.alloc_size;
                self.physical_handles.insert(handle.id(), handle.clone());
            }

            if allocated_size >= aligned_size {
                break;
            }
        }

        // Still need more handles.
        while allocated_size < aligned_size {
            let handle = storage.allocate(self.alloc_size)?;
            physical_handles.push(handle.clone());
            allocated_size += self.alloc_size;
            self.physical_handles.insert(handle.id(), handle.clone());
        }

        Ok(physical_handles)
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
        let page = VirtualMemoryPage::new(page_size);
        self.pages.insert(storage_id, page);

        self.ring.push_page(storage_id);

        Ok(storage_id)
    }

    /// Create a slice on a page
    fn create_slice(&self, offset: u64, size: u64, storage_id: StorageId) -> Slice {
        assert_eq!(
            offset % self.alloc_size,
            0,
            "Slice offset must be aligned to {}",
            self.alloc_size
        );

        let handle = SliceHandle::new();
        let storage = StorageHandle {
            id: storage_id,
            utilization: StorageUtilization { offset, size },
        };
        let padding = calculate_padding(size, self.alloc_size);

        Slice::new(storage, handle, padding)
    }

    /// Completely compact a page by merging adjacent free slices.
    /// Explanation of the page compaction algorithm.
    /// The merged signal indicates whether the last merge attempt succeeded.
    /// The outer loop tries to merge two slices of the page at each iteration.
    /// It will stop once one merge fails (no slices to merge) or if the length of the page has shrunk to one (completely compact).
    /// At each merge attempt, the slices layout inside the page might have changed.
    /// Therefore we need to restart the inner loop.
    fn compact_page(&mut self, page_id: &StorageId) -> bool {
        let mut merged = true;
        if let Some(page) = self.pages.get_mut(page_id) {
            while page.slices.len() > 1 && merged {
                merged = false;
                let page_addresses: Vec<u64> = page.slices.keys().cloned().collect();

                for address in page_addresses.iter() {
                    let slice_id = page.find_slice(*address).unwrap();
                    if let Some(slice) = self.virtual_slices.get(&slice_id)
                        && slice.is_free()
                        && page.merge_with_next_slice(*address, &mut self.virtual_slices)
                    {
                        merged = true;
                        break;
                    }
                }
            }
        }
        merged
    }

    /// Attempt to reserve an slice of an specific size
    /// Will search for the first free slice that has enough size and split it if necessary.
    /// If the slice is not mapped, it will map it automatically.
    ///
    /// I contrast to other pools, in this pool we need the storage in order to reserve.
    /// Therefore the workaround is to return always None in the try_reserve of the memory pool trait impl and implement the reservation strategy in the alloc method.
    pub fn try_reserve_with_storage<Storage: ComputeStorage>(
        &mut self,
        size: u64,
        storage: &mut Storage,
    ) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alloc_size);
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
        // Note that as free slices from very old pages are unmapped during defragmentation to reuse their handles, we might see here that the slice is not mapped, so just in case we map it if necessary to ensure memory is always valid.
        // CASE A: COMPLETELY UNMAPPED SLICE
        // Try allocate enough physical handles, popping from the freelist if available.
        // As all physical handles are the same size, computing the offset to map for each handle is straightforward.
        let physical_handles = if !old_slice.is_mapped() {
            let mut physical_handles = self.get_or_alloc_physical(storage, effective_size).ok()?;
            let mut offset = old_slice_offset;

            for physical in physical_handles.iter_mut() {
                storage.map(id, offset, physical).ok()?;
                offset += self.alloc_size;
            }

            Some(physical_handles)

        // CASE B: PARTIALLY MAPPED SLICE.
        // This case is similar to the previous one, with the difference that we need to map just as much handles as needed.
        } else if old_slice.mapped_size() < effective_size {
            let mut handles = old_slice.physical_handles.clone().unwrap();
            let size_to_map = effective_size - old_slice.mapped_size();
            let mut offset = old_slice_offset;
            let mut physical_handles = self.get_or_alloc_physical(storage, size_to_map).ok()?;

            for physical in physical_handles.iter_mut() {
                storage.map(id, offset, physical).ok()?;
                offset += self.alloc_size;
            }

            handles.extend(physical_handles);
            Some(handles)
        }
        // BEST CASE: COMPLETELY MAPPED:
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

    // Reset the main tracking data structures.
    #[cfg(test)]
    pub fn reset_tracking(&mut self) {
        self.recently_added_pages.clear();
        self.recently_allocated_size = 0;
        self.first_page = None;
    }

    #[cfg(test)]
    pub fn reusable_memory(&self) -> u64 {
        self.free_physical_queue.len() as u64 * self.alloc_size
    }

    /// This method implements the main defragmentation algorithm of the memory spaces.
    /// Compacts pages, collects free slices and unmaps them from their physical handles.
    /// The handles then become available for use when [`try_reserve`] or [`alloc`] come in.
    pub fn defragment<Storage: VirtualStorage>(&mut self, explicit: bool, storage: &mut Storage) {
        // Collect all free slices.
        let free_slice_ids: Vec<SliceId> = self
            .virtual_slices
            .iter() // Every call to defragment, all non recently added pages will be cleaned up.
            // If a page was recently added you might not want to free it.
            // Therefore, the list of recently added pages is cleared after you call defragment, so that in the next period any slice that is not referenced anymore is released.
            .filter(|(_, vs)| {
                vs.is_free() && (!self.recently_added_pages.contains(&vs.storage_id()) || explicit)
            })
            .map(|(id, _)| *id)
            .collect();

        if free_slice_ids.is_empty() {
            return;
        }

        // Unmap all free slices, releasing the physical handles for reuse.
        // The physical handles will be automatically reused on any [`alloc`] or [`try_reserve`]
        // calls.
        for slice_id in &free_slice_ids {
            if let Some(slice) = self.virtual_slices.get_mut(slice_id) {
                // Check if we can unmap the slice and release the handle if so.
                if let Some(handles) = slice.unmap() {
                    for mut handle in handles {
                        storage.unmap(slice.storage_id(), slice.storage_offset(), &mut handle);
                        // Will be reused on alloc or reserve.
                        self.free_physical_queue.push_back(handle.id());
                    }
                }
            }
        }

        // First merge all pages
        // This is an optimization. Not sure if will work with all backends, but when available it is worth trying
        if let Some(first) = self.first_page {
            let mut current = first;

            while let Some(next) = self.pages.get(&current).unwrap().next_page().cloned() {
                // Query the storage to check if they are aligned.
                // If pages are aligned we should be able to merge them without releasing the virtual memory space.
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
                    self.ring.remove_page(&next);

                    // Continue with the same current page since we merged the next one
                    continue;
                }

                current = next;
            }
        }

        // Finally, we should be able to deallocate pages that have become full free and are not 'pinned'
        // By pinned I meant they are not tracked as a recent allocation.
        //
        // Here ,I am collecting the pages that are either empty or completely free (all slices are free) and are not in the tracking of recent allocations list.
        // If explicit is true, we deallocate everything that is free even if it does appear on the deallocation list.
        let free_pages: Vec<StorageId> = self
            .pages
            .iter()
            .filter(|(id, p)| {
                (p.is_empty() || p.is_completely_free(&self.virtual_slices))
                    && (!self.recently_added_pages.contains(*id) || explicit)
            })
            .map(|(id, _p)| id)
            .cloned()
            .collect();

        // As the pages linked list will only work if the driver supports contiguous allocations, we follow this approach to ensure merging.
        // 1. Remove completely free pages and collect their slices.
        let mut orphaned_slices: Vec<SliceId> = Vec::new();
        let mut total_size = 0;
        for page_id in free_pages.iter() {
            if let Some(page) = self.pages.remove(page_id) {
                storage.free(*page_id);
                self.ring.remove_page(page_id);
                let keys: Vec<SliceId> = page.slices.values().cloned().collect();
                orphaned_slices.extend(keys);
                total_size += page.size;
            }
        }

        //dbg!(&orphaned_slices);

        // 2. Create a new page.
        if let Ok(new_page) =
            self.create_virtual_page(storage, total_size, self.last_allocated_page)
        {
            self.last_allocated_page = Some(new_page);

            let mut offset = 0;
            // Insert all orphaned slices into this page, at the carefully computed offsets.
            for slice_id in orphaned_slices.into_iter() {
                if let Some(page) = self.pages.get_mut(&new_page) {
                    page.insert_slice(offset, slice_id);
                }

                // Update the internal storage id
                if let Some(slice_mut) = self.virtual_slices.get_mut(&slice_id) {
                    slice_mut.slice.storage.id = new_page;
                    slice_mut.slice.storage.utilization.offset = offset;
                    offset += slice_mut.slice.storage.utilization.size;
                }
            }
        } else {
            // Page creation failed, therefore we have to remove orphaned slices.
            // There is no problem on that, though, as slices are just a software representation of a fragment of memory.
            for slice_id in orphaned_slices {
                self.virtual_slices.remove(&slice_id);
            }
        }

        // At this point, pages have been merged into a single one big chunk.
        // Therefore we can now try to compact them, merging all contiguous free slices.
        // In the sliced pool, you could only compact at a single page level, therefore the maximum compactability you could achieve if of size [`page_size`].
        // As free slices have been unmapped previously, merging them should be safe.
        // Anyway, in the [`try_reserve_with_storage`] function we are preventing any possible case.
        let ids: Vec<StorageId> = self.pages.keys().cloned().collect();
        for id in ids {
            self.compact_page(&id);
        }
    }
}

impl MemoryPool for VirtualMemoryPool {
    fn max_alloc_size(&self) -> u64 {
        self.alloc_size * 10
    }

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.virtual_slices
            .get(binding.id())
            .map(|vs| &vs.slice.storage)
    }

    /// Always returns None, since the reservation strategy is implemented in alloc.
    fn try_reserve(&mut self, _size: u64) -> Option<SliceHandle> {
        None
    }

    /// Directly allocate a new slice using the backing storage.
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        if let Some(handle) = self.try_reserve_with_storage(size, storage) {
            return Ok(handle);
        };

        let padding = calculate_padding(size, self.alloc_size);
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
        if let Some(last_allocation) = self.last_allocated_page
            && let Some(last_page) = self.pages.get_mut(&last_allocation)
        {
            last_page.next_page_id = Some(page_id);
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
        let mut offset = slice_offset;

        // Get physical backing
        let mut physical_handles = self.get_or_alloc_physical(storage, effective_size)?;

        for physical in physical_handles.iter_mut() {
            // Map physical to virtual
            storage.map(page_id, offset, physical)?;
            offset += self.alloc_size;
        }

        // Create virtual slice
        let virtual_slice = VirtualSlice::new(slice, Some(physical_handles));
        self.virtual_slices.insert(slice_id, virtual_slice);

        // Update page, inserting the slice at the correct offset.
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.insert_slice(slice_offset, slice_id);
        }
        // We do not need to create an extra slice, as the created slice currently is of the same size as the page.
        Ok(slice_handle)
    }

    /// Collect memory usage stats.
    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .virtual_slices
            .values()
            .filter(|vs| !vs.is_free())
            .collect();

        let total_reserved_size = self.physical_handles.iter().map(|(_, ph)| ph.size()).sum();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|vs| vs.slice.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|vs| vs.slice.padding).sum(),
            bytes_reserved: total_reserved_size,
        }
    }

    /// Clean up all physical memory handles that have become free after defragmentation.
    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        // Defragment and reset tracking every dealloc_period allocations.
        if explicit || alloc_nr % self.dealloc_period == 0 {
            self.defragment(explicit, storage);
            self.recently_added_pages.clear();
            self.recently_allocated_size = 0;
            self.first_page = None;
        }

        // If explicitly called, also release physical memory to make up space.
        if explicit {
            for id in self.free_physical_queue.drain(..) {
                self.physical_handles.remove(&id);
                storage.release(id)
            }
        }
    }
}
