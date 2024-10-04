use super::{MemoryPoolBinding, MemoryPoolHandle, SliceHandle, SliceId};
use crate::{
    memory_management::MemoryLock,
    storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization},
};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// A memory pool that allocates fixed-size chunks (32 bytes each) and reuses them to minimize allocations.
///
/// - Only one slice is supported per chunk due to the limitations in WGPU where small allocations cannot be offset.
/// - The pool uses a ring buffer to efficiently manage and reuse chunks.
///
/// Fields:
/// - `chunks`: A hashmap storing the allocated chunks by their IDs.
/// - `slices`: A hashmap storing the slices by their IDs.
/// - `ring_buffer`: A vector used as a ring buffer to manage chunk reuse.
/// - `index`: The current position in the ring buffer.
pub struct SmallMemoryPool {
    chunks: HashMap<StorageId, SmallChunk>,
    slices: HashMap<SliceId, SmallSlice>,
    ring_buffer: Vec<StorageId>,
    index: usize,
    buffer_storage_alignment_offset: usize,
}

#[derive(new, Debug)]
pub struct SmallChunk {
    pub slice: Option<SliceId>,
}

#[derive(new, Debug)]
pub struct SmallSlice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: usize,
}

impl SmallSlice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

impl SmallMemoryPool {
    pub fn new(buffer_storage_alignment_offset: usize) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            ring_buffer: Vec::new(),
            index: 0,
            buffer_storage_alignment_offset,
        }
    }

    /// Returns the resource from the storage, for the specified handle.
    pub fn get(&self, binding: &MemoryPoolBinding) -> Option<&StorageHandle> {
        self.slices.get(binding.slice.id()).map(|s| &s.storage)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        locked: Option<&MemoryLock>,
    ) -> MemoryPoolHandle {
        assert!(size <= self.buffer_storage_alignment_offset);
        let slice = self.get_free_slice(size, locked);

        match slice {
            Some(slice) => MemoryPoolHandle {
                slice: slice.clone(),
            },
            None => self.alloc(storage, size),
        }
    }

    pub fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> MemoryPoolHandle {
        assert!(size <= self.buffer_storage_alignment_offset);

        self.alloc_slice(storage, size)
    }

    fn alloc_slice<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        slice_size: usize,
    ) -> MemoryPoolHandle {
        let storage_id = self.create_chunk(storage, self.buffer_storage_alignment_offset);
        let slice = self.allocate_slice(storage_id, slice_size);

        let handle_slice = slice.handle.clone();
        self.update_chunk_metadata(slice);

        MemoryPoolHandle {
            slice: handle_slice,
        }
    }

    fn allocate_slice(&self, storage_id: StorageId, slice_size: usize) -> SmallSlice {
        let slice = self.create_slice(0, slice_size, storage_id);

        let effective_size = slice.effective_size();
        assert_eq!(effective_size, self.buffer_storage_alignment_offset);

        slice
    }

    fn update_chunk_metadata(&mut self, slice: SmallSlice) {
        let slice_id = *slice.handle.id();

        self.chunks.get_mut(&slice.storage.id).unwrap().slice = Some(slice_id);
        self.slices.insert(slice_id, slice);
    }

    fn find_free_slice(&mut self, locked: Option<&MemoryLock>) -> Option<SliceId> {
        for _ in 0..self.ring_buffer.len() {
            let storage_id = self.ring_buffer.get(self.index).unwrap();
            if let Some(locked) = locked.as_ref() {
                if locked.is_locked(storage_id) {
                    continue;
                }
            }
            let chunk = self.chunks.get(storage_id).unwrap();
            let slice = self.slices.get(&chunk.slice.unwrap()).unwrap();
            self.index = (self.index + 1) % self.ring_buffer.len();
            if slice.handle.is_free() {
                return Some(*slice.handle.id());
            }
        }
        None
    }

    /// Finds a free slice that can contain the given size
    /// Returns the chunk's id and size.
    fn get_free_slice(&mut self, size: usize, locked: Option<&MemoryLock>) -> Option<SliceHandle> {
        let slice_id = self.find_free_slice(locked)?;

        let slice = self.slices.get_mut(&slice_id).unwrap();
        let old_slice_size = slice.effective_size();

        let offset = match slice.storage.utilization {
            StorageUtilization::Full(_) => 0,
            StorageUtilization::Slice { offset, size: _ } => offset,
        };
        assert_eq!(offset, 0);
        slice.storage.utilization = StorageUtilization::Slice { offset, size };
        let new_padding = old_slice_size - size;
        slice.padding = new_padding;
        assert_eq!(
            slice.effective_size(),
            old_slice_size,
            "new and old slice should have the same size"
        );

        Some(slice.handle.clone())
    }

    /// Creates a slice of size `size` upon the given chunk with the given offset.
    fn create_slice(&self, offset: usize, size: usize, storage_id: StorageId) -> SmallSlice {
        assert_eq!(offset, 0);
        let handle = SliceHandle::new();

        let storage = StorageHandle {
            id: storage_id,
            utilization: StorageUtilization::Slice { offset, size },
        };

        let padding = calculate_padding(size, self.buffer_storage_alignment_offset);

        SmallSlice::new(storage, handle, padding)
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> StorageId {
        let padding = calculate_padding(size, self.buffer_storage_alignment_offset);
        let effective_size = size + padding;

        let storage = storage.alloc(effective_size);
        let id = storage.id;
        self.ring_buffer.push(id);
        self.chunks.insert(id, SmallChunk::new(None));
        id
    }

    #[allow(unused)]
    fn deallocate<Storage: ComputeStorage>(&mut self, _storage: &mut Storage) {
        todo!()
    }
}

fn calculate_padding(size: usize, buffer_storage_alignment_offset: usize) -> usize {
    let remainder = size % buffer_storage_alignment_offset;
    if remainder != 0 {
        buffer_storage_alignment_offset - remainder
    } else {
        0
    }
}
