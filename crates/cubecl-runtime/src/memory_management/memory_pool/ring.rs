use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::storage::StorageId;

use super::{MemoryChunk, MemoryFragment};

#[derive(Debug)]
pub struct RingBuffer {
    queue: Vec<StorageId>,
    chunk_positions: HashMap<StorageId, usize>,
    cursor_slice: u64,
    cursor_chunk: usize,
    buffer_alignment: u64,
}

impl RingBuffer {
    pub fn new(buffer_alignment: u64) -> Self {
        Self {
            queue: Vec::new(),
            chunk_positions: HashMap::new(),
            cursor_slice: 0,
            cursor_chunk: 0,
            buffer_alignment,
        }
    }

    pub fn push_page(&mut self, storage_id: StorageId) {
        self.queue.push(storage_id);
        self.chunk_positions
            .insert(storage_id, self.queue.len() - 1);
    }

    pub fn find_free_slice<M: MemoryChunk>(
        &mut self,
        size: u64,
        pages: &mut HashMap<StorageId, M>,
        slices: &mut HashMap<M::Key, M::Fragment>,
    ) -> Option<M::Key> {
        let max_second = self.cursor_chunk;
        let result = self.find_free_slice_in_all_chunks(size, pages, slices, self.queue.len());

        if result.is_some() {
            return result;
        }

        self.cursor_chunk = 0;
        self.cursor_slice = 0;
        self.find_free_slice_in_all_chunks(size, pages, slices, max_second)
    }

    fn find_free_slice_in_chunk<M: MemoryChunk>(
        &mut self,
        size: u64,
        page: &mut M,
        slices: &mut HashMap<M::Key, M::Fragment>,
        mut slice_index: u64,
    ) -> Option<(u64, M::Key)> {
        while let Some(slice_id) = page.find_slice(slice_index) {
            //mutable borrow scope
            {
                let slice = slices.get_mut(&slice_id).unwrap();

                let is_big_enough = slice.effective_size() >= size;
                let is_free = slice.is_free();

                if is_big_enough && is_free {
                    if slice.effective_size() > size
                        && let Some(new_slice) = slice.split(size, self.buffer_alignment)
                    {
                        let new_slice_id = new_slice.id();
                        page.insert_slice(slice.next_slice_position(), new_slice_id);
                        slices.insert(new_slice_id, new_slice);
                    }
                    return Some((slice_index, slice_id));
                }
            }
            {
                let slice = slices.get_mut(&slice_id).unwrap();
                let is_free = slice.is_free();
                if is_free && page.merge_with_next_slice(slice_index, slices) {
                    continue;
                }
            }

            if let Some(slice) = slices.get(&slice_id) {
                slice_index = slice.next_slice_position();
            } else {
                panic!("current slice_id should still be valid after potential merge");
            }
        }

        None
    }

    fn find_free_slice_in_all_chunks<M: MemoryChunk>(
        &mut self,
        size: u64,
        pages: &mut HashMap<StorageId, M>,
        slices: &mut HashMap<M::Key, M::Fragment>,
        max_cursor_position: usize,
    ) -> Option<M::Key> {
        let start = self.cursor_chunk;
        let end = usize::min(self.queue.len(), max_cursor_position);
        let mut slice_index = self.cursor_slice;

        for chunk_index in start..end {
            if chunk_index > start {
                slice_index = 0;
            }

            if let Some(id) = self.queue.get(chunk_index) {
                let chunk = pages.get_mut(id).unwrap();
                let result = self.find_free_slice_in_chunk(size, chunk, slices, slice_index);

                if let Some((_cursor_slice, slice)) = result {
                    let slice = slices.get(&slice).unwrap();
                    self.cursor_slice = slice.next_slice_position();
                    self.cursor_chunk = chunk_index;
                    return Some(slice.id());
                }
            }
            self.cursor_chunk = chunk_index;
            self.cursor_slice = 0;
        }

        None
    }

    /// Remove a page from the ring buffer
    pub fn remove_page(&mut self, storage_id: &StorageId) -> bool {
        if let Some(&position) = self.chunk_positions.get(storage_id) {
            // Remove from the queue
            self.queue.remove(position);

            // Remove from positions map
            self.chunk_positions.remove(storage_id);

            // Update positions for all pages that came after the removed one
            for (_id, pos) in self.chunk_positions.iter_mut() {
                if *pos > position {
                    *pos -= 1;
                }
            }

            // Adjust cursors if necessary
            if self.cursor_chunk > position {
                // The current cursor chunk moved one position back
                self.cursor_chunk -= 1;
            } else if self.cursor_chunk == position {
                // We removed the page we were currently on
                if self.cursor_chunk >= self.queue.len() {
                    // We were on the last page, wrap around or reset
                    self.cursor_chunk = if self.queue.is_empty() {
                        0
                    } else {
                        self.queue.len() - 1
                    };
                    self.cursor_slice = 0;
                }
                // If cursor_chunk < queue.len(), we can stay on the same index
                // as it now points to the next page
                self.cursor_slice = 0;
            }

            true // Successfully removed
        } else {
            false // Page not found
        }
    }

    /// Check if a page exists in the ring buffer
    pub fn contains_page(&self, storage_id: &StorageId) -> bool {
        self.chunk_positions.contains_key(storage_id)
    }

    /// Get the current number of pages in the ring buffer
    pub fn page_count(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::memory_management::SliceId;
    use crate::memory_management::memory_pool::Slice;
    use crate::{
        memory_management::memory_pool::{MemoryPage, SliceHandle},
        storage::StorageHandle,
    };

    use super::*;

    #[test]
    fn simple_1() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 200]);

        ring.push_page(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        let slice = ring.find_free_slice(50, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, slice_ids[0]);
        assert_eq!(slices.get(&slice).unwrap().effective_size(), 50);
        assert_eq!(slices.len(), 3);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 3);
    }

    #[test]
    fn simple_2() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 200]);

        ring.push_page(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        let slice = ring.find_free_slice(150, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, slice_ids[0]);
        assert_eq!(slices.get(&slice).unwrap().effective_size(), 150);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 2);
    }

    #[test]
    fn multiple_chunks() {
        let mut ring = RingBuffer::new(1);

        let (storage_id_1, mut slice_ids, mut slices, chunk_1) = new_chunk(&[100, 200]);
        let (storage_id_2, slice_ids_2, slices_2, chunk_2) = new_chunk(&[200, 200]);

        ring.push_page(storage_id_1);
        ring.push_page(storage_id_2);

        let mut chunks = HashMap::from([(storage_id_1, chunk_1), (storage_id_2, chunk_2)]);

        slice_ids.extend(slice_ids_2);
        slices.extend(slices_2);

        // Clone references to control what slice is free:
        let _slice_1 = slices.get(&slice_ids[1]).unwrap().handle.clone();
        let _slice_3 = slices.get(&slice_ids[3]).unwrap().handle.clone();

        let slice = ring.find_free_slice(200, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, slice_ids[2]);

        let slice = ring.find_free_slice(100, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, slice_ids[0]);
    }

    #[test]
    fn find_free_slice_with_exact_fit() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 200]);

        ring.push_page(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        // Clone reference to control what slice is free:
        let _slice_1 = slices.get(&slice_ids[0]).unwrap().handle.clone();

        let slice = ring.find_free_slice(200, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, slice_ids[1]);
        assert_eq!(slices.get(&slice).unwrap().effective_size(), 200);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 2);
    }

    #[test]
    fn find_free_slice_with_merging() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 50, 100]);

        ring.push_page(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        let slice = ring.find_free_slice(250, &mut chunks, &mut slices).unwrap();

        assert_eq!(slice, slice_ids[0]);
        assert_eq!(slices.get(&slice).unwrap().effective_size(), 250);
        assert_eq!(slices.len(), 1);
        assert_eq!(chunks.values().last().unwrap().slices.len(), 1);
    }

    #[test]
    fn find_free_slice_with_multiple_chunks_and_merging() {
        let mut ring = RingBuffer::new(1);

        let (storage_id_1, mut slice_ids, mut slices, page_1) = new_chunk(&[50, 50]);
        let (storage_id_2, slice_ids_2, slices_2, page_2) = new_chunk(&[100, 50]);
        slice_ids.extend(slice_ids_2);
        slices.extend(slices_2);

        ring.push_page(storage_id_1);
        ring.push_page(storage_id_2);

        let mut pages = HashMap::from([(storage_id_1, page_1), (storage_id_2, page_2)]);

        let slice = ring.find_free_slice(150, &mut pages, &mut slices).unwrap();

        assert_eq!(slices.get(&slice).unwrap().effective_size(), 150);
        assert_eq!(slices.len(), 2);
        assert_eq!(pages.values().last().unwrap().slices.len(), 1);
    }

    fn new_chunk(
        slice_sizes: &[u64],
    ) -> (StorageId, Vec<SliceId>, HashMap<SliceId, Slice>, MemoryPage) {
        let offsets: Vec<_> = slice_sizes
            .iter()
            .scan(0, |state, size| {
                let offset = *state;
                *state += *size;
                Some(offset)
            })
            .collect();

        let storage_id = StorageId::new();

        let slices: Vec<_> = offsets
            .iter()
            .zip(slice_sizes)
            .map(|(&offset, &size)| Slice {
                storage: StorageHandle {
                    id: storage_id,
                    utilization: crate::storage::StorageUtilization { offset, size },
                },
                handle: SliceHandle::new(),
                padding: 0,
            })
            .collect();

        let mem_page = MemoryPage {
            slices: slices
                .iter()
                .zip(offsets)
                .map(|(slice, offset)| (offset, slice.id()))
                .collect(),
        };

        (
            storage_id,
            slices.iter().map(|slice| slice.id()).collect(),
            slices
                .into_iter()
                .map(|slice| (slice.id(), slice))
                .collect(),
            mem_page,
        )
    }

    /// Added this test to verify that page removal works properly.
    #[test]
    fn test_ring_buffer_page_removal() {
        let mut ring = RingBuffer::new(1);

        // Create three chunks with different slice configurations
        let (storage_id_1, slice_ids_1, mut slices, chunk_1) = new_chunk(&[100, 200]);
        let (storage_id_2, slice_ids_2, slices_2, chunk_2) = new_chunk(&[150, 250]);
        let (storage_id_3, slice_ids_3, slices_3, chunk_3) = new_chunk(&[300]);

        // Add all pages to ring buffer
        ring.push_page(storage_id_1);
        ring.push_page(storage_id_2);
        ring.push_page(storage_id_3);

        let mut chunks = HashMap::from([
            (storage_id_1, chunk_1),
            (storage_id_2, chunk_2),
            (storage_id_3, chunk_3),
        ]);

        // Merge all slices
        slices.extend(slices_2);
        slices.extend(slices_3);

        // Verify initial state
        assert_eq!(ring.page_count(), 3);
        assert!(ring.contains_page(&storage_id_1));
        assert!(ring.contains_page(&storage_id_2));
        assert!(ring.contains_page(&storage_id_3));

        // Test allocation before removal
        let slice_before = ring.find_free_slice(100, &mut chunks, &mut slices).unwrap();
        assert!(slices.contains_key(&slice_before));

        // Remove the middle page
        assert!(ring.remove_page(&storage_id_2));
        chunks.remove(&storage_id_2);

        // Remove slices that belonged to the removed page
        let slices_to_remove: Vec<SliceId> = slices
            .iter()
            .filter(|(_, slice)| slice.storage.id == storage_id_2)
            .map(|(&id, _)| id)
            .collect();

        for slice_id in slices_to_remove {
            slices.remove(&slice_id);
        }

        // Verify removal
        assert_eq!(ring.page_count(), 2);
        assert!(ring.contains_page(&storage_id_1));
        assert!(!ring.contains_page(&storage_id_2));
        assert!(ring.contains_page(&storage_id_3));

        // Verify positions are updated correctly
        assert_eq!(ring.chunk_positions.get(&storage_id_1), Some(&0));
        assert_eq!(ring.chunk_positions.get(&storage_id_3), Some(&1));

        // Test that allocation still works after removal
        let slice_after = ring.find_free_slice(250, &mut chunks, &mut slices).unwrap();
        assert!(slices.contains_key(&slice_after));
        // Should find the slice from storage_id_3 (300 bytes)
        assert_eq!(slices.get(&slice_after).unwrap().storage.id, storage_id_3);

        // Remove first page
        assert!(ring.remove_page(&storage_id_1));
        chunks.remove(&storage_id_1);

        // Remove slices that belonged to the removed page
        let slices_to_remove: Vec<SliceId> = slices
            .iter()
            .filter(|(_, slice)| slice.storage.id == storage_id_1)
            .map(|(&id, _)| id)
            .collect();

        for slice_id in slices_to_remove {
            slices.remove(&slice_id);
        }

        // Verify only one page remains
        assert_eq!(ring.page_count(), 1);
        assert!(!ring.contains_page(&storage_id_1));
        assert!(ring.contains_page(&storage_id_3));
        assert_eq!(ring.chunk_positions.get(&storage_id_3), Some(&0));

        // Try to remove non-existent page
        assert!(!ring.remove_page(&storage_id_2));
        assert_eq!(ring.page_count(), 1);

        // Remove last page
        assert!(ring.remove_page(&storage_id_3));
        chunks.remove(&storage_id_3);

        assert_eq!(ring.page_count(), 0);
        assert!(ring.chunk_positions.is_empty());

        // Verify allocation fails when no pages exist
        let no_slice = ring.find_free_slice(100, &mut chunks, &mut slices);
        assert!(no_slice.is_none());
    }
}
