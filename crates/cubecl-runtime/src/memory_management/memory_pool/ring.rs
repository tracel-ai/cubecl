use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::{memory_management::MemoryLock, storage::StorageId};

use super::{Chunk, Slice, SliceId};

#[derive(Debug)]
pub struct RingBuffer {
    queue: Vec<StorageId>,
    chunk_positions: HashMap<StorageId, usize>,
    cursor_slice: usize,
    cursor_chunk: usize,
    buffer_alignment: usize,
}

impl RingBuffer {
    pub fn new(buffer_alignment: usize) -> Self {
        Self {
            queue: Vec::new(),
            chunk_positions: HashMap::new(),
            cursor_slice: 0,
            cursor_chunk: 0,
            buffer_alignment,
        }
    }

    pub fn push_chunk(&mut self, storage_id: StorageId) {
        self.queue.push(storage_id);
        self.chunk_positions
            .insert(storage_id, self.queue.len() - 1);
    }

    pub fn find_free_slice(
        &mut self,
        size: usize,
        chunks: &mut HashMap<StorageId, Chunk>,
        slices: &mut HashMap<SliceId, Slice>,
        locked: Option<&MemoryLock>,
    ) -> Option<SliceId> {
        let max_second = self.cursor_chunk;
        let result =
            self.find_free_slice_in_all_chunks(size, chunks, slices, self.queue.len(), locked);

        if result.is_some() {
            return result;
        }

        self.cursor_chunk = 0;
        self.cursor_slice = 0;
        self.find_free_slice_in_all_chunks(size, chunks, slices, max_second, locked)
    }

    fn find_free_slice_in_chunk(
        &mut self,
        size: usize,
        chunk: &mut Chunk,
        slices: &mut HashMap<SliceId, Slice>,
        mut slice_index: usize,
    ) -> Option<(usize, SliceId)> {
        while let Some(slice_id) = chunk.slice(slice_index) {
            //mutable borrow scope
            {
                let slice = slices.get_mut(&slice_id).unwrap();

                let is_big_enough = slice.size() >= size;
                let is_free = slice.is_free();

                if is_big_enough && is_free {
                    if slice.size() > size {
                        if let Some(new_slice) = slice.split(size, self.buffer_alignment) {
                            let new_slice_id = new_slice.id();
                            chunk.insert_slice(slice.next_slice_position(), new_slice, slices);
                            slices.get(&new_slice_id).unwrap();
                        }
                    }
                    return Some((slice_index, slice_id));
                }
            }
            {
                let slice = slices.get_mut(&slice_id).unwrap();
                let is_free = slice.is_free();
                if is_free && chunk.merge_next_slice(slice_index, slices) {
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

    fn find_free_slice_in_all_chunks(
        &mut self,
        size: usize,
        chunks: &mut HashMap<StorageId, Chunk>,
        slices: &mut HashMap<SliceId, Slice>,
        max_cursor_position: usize,
        locked: Option<&MemoryLock>,
    ) -> Option<SliceId> {
        let start = self.cursor_chunk;
        let end = usize::min(self.queue.len(), max_cursor_position);
        let mut slice_index = self.cursor_slice;

        for chunk_index in start..end {
            if chunk_index > start {
                slice_index = 0;
            }

            if let Some(id) = self.queue.get(chunk_index) {
                if let Some(locked) = locked.as_ref() {
                    if locked.is_locked(id) {
                        continue;
                    }
                }

                let chunk = chunks.get_mut(id).unwrap();
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
}

#[cfg(test)]
mod tests {
    use crate::{
        memory_management::memory_pool::{MemoryPage, SliceHandle},
        storage::StorageHandle,
    };

    use super::*;

    #[test]
    fn simple_1() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 200]);

        ring.push_chunk(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        let slice = ring
            .find_free_slice(50, &mut chunks, &mut slices, None)
            .unwrap();

        assert_eq!(slice, slice_ids[0]);
        assert_eq!(slices.get(&slice).unwrap().size(), 50);
        assert_eq!(slices.len(), 3);
        assert_eq!(chunks.values().last().unwrap().slices.slices.len(), 3);
    }

    #[test]
    fn simple_2() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 200]);

        ring.push_chunk(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        let slice = ring
            .find_free_slice(150, &mut chunks, &mut slices, None)
            .unwrap();

        assert_eq!(slice, slice_ids[0]);
        assert_eq!(slices.get(&slice).unwrap().size(), 150);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.slices.len(), 2);
    }

    #[test]
    fn multiple_chunks() {
        let mut ring = RingBuffer::new(1);

        let (storage_id_1, mut slice_ids, mut slices, chunk_1) = new_chunk(&[100, 200]);
        let (storage_id_2, slice_ids_2, slices_2, chunk_2) = new_chunk(&[200, 200]);

        ring.push_chunk(storage_id_1);
        ring.push_chunk(storage_id_2);

        let mut chunks = HashMap::from([(storage_id_1, chunk_1), (storage_id_2, chunk_2)]);

        slice_ids.extend(slice_ids_2);
        slices.extend(slices_2);

        // Clone references to control what slice is free:
        let _slice_1 = slices.get(&slice_ids[1]).unwrap().handle.clone();
        let _slice_3 = slices.get(&slice_ids[3]).unwrap().handle.clone();

        let slice = ring
            .find_free_slice(200, &mut chunks, &mut slices, None)
            .unwrap();

        assert_eq!(slice, slice_ids[2]);

        let slice = ring
            .find_free_slice(100, &mut chunks, &mut slices, None)
            .unwrap();

        assert_eq!(slice, slice_ids[0]);
    }

    #[test]
    fn find_free_slice_with_exact_fit() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 200]);

        ring.push_chunk(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        // Clone reference to control what slice is free:
        let _slice_1 = slices.get(&slice_ids[0]).unwrap().handle.clone();

        let slice = ring
            .find_free_slice(200, &mut chunks, &mut slices, None)
            .unwrap();

        assert_eq!(slice, slice_ids[1]);
        assert_eq!(slices.get(&slice).unwrap().size(), 200);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.slices.len(), 2);
    }

    #[test]
    fn find_free_slice_with_merging() {
        let mut ring = RingBuffer::new(1);

        let (storage_id, slice_ids, mut slices, chunk) = new_chunk(&[100, 50, 100]);

        ring.push_chunk(storage_id);
        let mut chunks = HashMap::from([(storage_id, chunk)]);

        let slice = ring
            .find_free_slice(250, &mut chunks, &mut slices, None)
            .unwrap();

        assert_eq!(slice, slice_ids[0]);
        assert_eq!(slices.get(&slice).unwrap().size(), 250);
        assert_eq!(slices.len(), 1);
        assert_eq!(chunks.values().last().unwrap().slices.slices.len(), 1);
    }

    #[test]
    fn find_free_slice_with_multiple_chunks_and_merging() {
        let mut ring = RingBuffer::new(1);

        let (storage_id_1, mut slice_ids, mut slices, chunk_1) = new_chunk(&[50, 50]);
        let (storage_id_2, slice_ids_2, slices_2, chunk_2) = new_chunk(&[100, 50]);
        slice_ids.extend(slice_ids_2);
        slices.extend(slices_2);

        ring.push_chunk(storage_id_1);
        ring.push_chunk(storage_id_2);

        let mut chunks = HashMap::from([(storage_id_1, chunk_1), (storage_id_2, chunk_2)]);

        let slice = ring
            .find_free_slice(150, &mut chunks, &mut slices, None)
            .unwrap();

        assert_eq!(slices.get(&slice).unwrap().size(), 150);
        assert_eq!(slices.len(), 2);
        assert_eq!(chunks.values().last().unwrap().slices.slices.len(), 1);
    }

    #[test]
    fn excludes_locked_storage() {
        let mut ring = RingBuffer::new(1);

        let (storage_id_1, mut slice_ids, mut slices, chunk_1) = new_chunk(&[100, 100]);
        let (storage_id_2, slice_ids_2, slices_2, chunk_2) = new_chunk(&[100, 100]);

        ring.push_chunk(storage_id_1);
        ring.push_chunk(storage_id_2);

        let mut chunks = HashMap::from([(storage_id_1, chunk_1), (storage_id_2, chunk_2)]);

        slice_ids.extend(slice_ids_2);
        slices.extend(slices_2);

        let slice = ring
            .find_free_slice(100, &mut chunks, &mut slices, None)
            .unwrap();
        assert_eq!(slice, slice_ids[0]);

        let mut locked = MemoryLock::default();
        locked.add_locked(storage_id_1);

        let slice = ring
            .find_free_slice(100, &mut chunks, &mut slices, Some(&locked))
            .unwrap();
        assert_eq!(slice, slice_ids[2]);
    }

    fn new_chunk(
        slice_sizes: &[usize],
    ) -> (StorageId, Vec<SliceId>, HashMap<SliceId, Slice>, Chunk) {
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
                    utilization: crate::storage::StorageUtilization::Slice { offset, size },
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

        let chunk = Chunk {
            alloc_size: 1024 * 1024, // Arbitrary, just pretend we have a big enough allocation.
            slices: mem_page,
        };

        (
            storage_id,
            slices.iter().map(|slice| slice.id()).collect(),
            slices
                .into_iter()
                .map(|slice| (slice.id(), slice))
                .collect(),
            chunk,
        )
    }
}
