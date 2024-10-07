use super::index::SearchIndex;
use super::{MemoryPoolBinding, MemoryPoolHandle, RingBuffer, SliceHandle, SliceId};
use crate::memory_management::MemoryLock;
use crate::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use alloc::vec::Vec;
use hashbrown::HashMap;

pub struct MemoryPool {
    chunks: HashMap<StorageId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    #[allow(unused)] // will be used when we rewrite memory extension
    memory_extension_strategy: MemoryExtensionStrategy,
    rounding: RoundingStrategy,
    storage_index: SearchIndex<StorageId>,
    ring: RingBuffer,
    recently_added_chunks: Vec<StorageId>,
    recently_allocated_size: usize,
    buffer_alignment: usize,
}

#[derive(new, Debug)]
pub struct Chunk {
    pub alloc_size: usize,
    pub slices: MemoryPage,
}

// TODO: consider using generic trait and decouple from Slice
#[derive(new, Debug)]
pub struct MemoryPage {
    pub slices: HashMap<usize, SliceId>,
}

impl MemoryPage {
    /// merge slice at first_slice_address with the next slice (if there is one and if it's free)
    /// return a boolean representing if a merge happened
    fn merge_with_next_slice(
        &mut self,
        first_slice_address: usize,
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
                first_slice.storage.utilization = StorageUtilization::Slice {
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

    fn find_slice(&self, address: usize) -> Option<SliceId> {
        let slice_id = self.slices.get(&address);
        slice_id.copied()
    }

    fn insert_slice(&mut self, address: usize, slice: SliceId) {
        self.slices.insert(address, slice);
    }
}

#[derive(new, Debug)]
pub struct Slice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: usize,
}

impl Slice {
    pub fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }
}

const MIN_SIZE_NEEDED_TO_OFFSET: usize = 16;

pub enum RoundingStrategy {
    FixedAmount(usize),
    #[allow(unused)]
    None,
}

impl RoundingStrategy {
    fn alloc_size(&self, size: usize) -> usize {
        match self {
            RoundingStrategy::FixedAmount(chunk_size) => {
                assert!(*chunk_size >= size);
                *chunk_size
            }
            RoundingStrategy::None => size,
        }
    }
}

/// The strategy defines the frequency at which merging of free slices (defragmentation) occurs
#[allow(unused)]
#[derive(Debug)]
pub enum MemoryExtensionStrategy {
    /// Once every n calls to reserve.
    PeriodTick {
        /// Number of calls to be executed before triggering the defragmentation.
        period: usize,
        /// Current state. Should start at zero.
        state: usize,
    },
    /// Never defragment.
    Never,
}

#[allow(unused)]
impl MemoryExtensionStrategy {
    /// Create a new strategy with the given period.
    pub fn new_period_tick(period: usize) -> Self {
        MemoryExtensionStrategy::PeriodTick { period, state: 0 }
    }

    #[allow(unused)]
    fn should_extend_max_memory(&mut self) -> bool {
        match self {
            MemoryExtensionStrategy::PeriodTick { period, state } => {
                *state = (*state + 1) % *period;
                *state == 0
            }
            MemoryExtensionStrategy::Never => false,
        }
    }
}

impl MemoryPool {
    pub fn new(
        merging_strategy: MemoryExtensionStrategy,
        alloc_strategy: RoundingStrategy,
        buffer_alignment: usize,
    ) -> Self {
        Self {
            chunks: HashMap::new(),
            slices: HashMap::new(),
            memory_extension_strategy: merging_strategy,
            rounding: alloc_strategy,
            storage_index: SearchIndex::new(),
            ring: RingBuffer::new(buffer_alignment),
            recently_added_chunks: Vec::new(),
            recently_allocated_size: 0,
            buffer_alignment,
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
        let alloc_size = self.rounding.alloc_size(size);
        self.alloc_slice(storage, alloc_size, size)
    }

    fn alloc_slice<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_size: usize,
        slice_size: usize,
    ) -> MemoryPoolHandle {
        let chunk_size = self.rounding.alloc_size(alloc_size);
        let storage_id = self.create_chunk(storage, chunk_size);
        let chunk_size = self.chunks.get(&storage_id).unwrap().alloc_size;
        self.recently_added_chunks.push(storage_id);
        self.recently_allocated_size += chunk_size;

        let (slice, extra_slice) = self.allocate_slices(storage_id, chunk_size, slice_size);

        let handle_slice = slice.handle.clone();
        self.update_chunk_metadata(slice, extra_slice);

        MemoryPoolHandle {
            slice: handle_slice,
        }
    }

    fn allocate_slices(
        &self,
        storage_id: StorageId,
        alloc_size: usize,
        slice_size: usize,
    ) -> (Slice, Option<Slice>) {
        let slice = self.create_slice(0, slice_size, storage_id);

        let effective_size = slice.effective_size();

        let extra_slice = if effective_size < alloc_size {
            Some(self.create_slice(effective_size, alloc_size - effective_size, storage_id))
        } else {
            None
        };

        (slice, extra_slice)
    }

    fn update_chunk_metadata(&mut self, slice: Slice, extra_slice: Option<Slice>) {
        let storage_id = slice.storage.id;
        let slice_id = *slice.handle.id();
        let slice_offset = slice.storage.offset();

        self.slices.insert(slice_id, slice);
        let chunk = self.chunks.get_mut(&storage_id).unwrap();
        chunk.slices.slices.insert(slice_offset, slice_id);

        if let Some(extra_slice) = extra_slice {
            let extra_slice_id = *extra_slice.handle.id();
            let extra_slice_offset = extra_slice.storage.offset();
            self.slices.insert(extra_slice_id, extra_slice);
            chunk
                .slices
                .slices
                .insert(extra_slice_offset, extra_slice_id);
        }
    }

    #[allow(unused)]
    fn display_memory_usage(&self) {
        let total_memory_usage: f64 = self
            .chunks
            .values()
            .map(|chunk| chunk.alloc_size as f64)
            .sum();
        let effective_memory_usage: f64 = self
            .slices
            .values()
            .filter(|slice| slice.handle.is_free())
            .map(|slice| slice.storage.size() as f64)
            .sum();
        let ratio = 100.0 * effective_memory_usage / total_memory_usage;
        log::info!("the memory usage is {ratio}");
    }

    /// Finds a free slice that can contain the given size
    /// Returns the chunk's id and size.
    fn get_free_slice(&mut self, size: usize, locked: Option<&MemoryLock>) -> Option<SliceHandle> {
        if size < MIN_SIZE_NEEDED_TO_OFFSET {
            return None;
        }

        let padding = calculate_padding(size, self.buffer_alignment);
        let effective_size = size + padding;

        let slice_id = self.ring.find_free_slice(
            effective_size,
            &mut self.chunks,
            &mut self.slices,
            locked,
        )?;

        let slice = self.slices.get_mut(&slice_id).unwrap();
        let old_slice_size = slice.effective_size();

        let offset = match slice.storage.utilization {
            StorageUtilization::Full(_) => 0,
            StorageUtilization::Slice { offset, size: _ } => offset,
        };
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
    fn create_slice(&self, offset: usize, size: usize, storage_id: StorageId) -> Slice {
        assert_eq!(
            offset % self.buffer_alignment,
            0,
            "slice with offset {offset} needs to be a multiple of {}",
            self.buffer_alignment
        );
        if offset > 0 && size < MIN_SIZE_NEEDED_TO_OFFSET {
            panic!("tried to create slice of size {size} with an offset while the size needs to atleast be of size {MIN_SIZE_NEEDED_TO_OFFSET} for offset support");
        }
        let handle = SliceHandle::new();

        let storage = StorageHandle {
            id: storage_id,
            utilization: StorageUtilization::Slice { offset, size },
        };

        let padding = calculate_padding(size, self.buffer_alignment);

        Slice::new(storage, handle, padding)
    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
    ) -> StorageId {
        let padding = calculate_padding(size, self.buffer_alignment);
        let effective_size = size + padding;

        let storage = storage.alloc(effective_size);

        let id = storage.id;
        self.ring.push_chunk(id);

        self.chunks.insert(
            id,
            Chunk::new(effective_size, MemoryPage::new(HashMap::new())),
        );
        self.storage_index.insert(id, size);

        id
    }
}

fn calculate_padding(size: usize, buffer_alignment: usize) -> usize {
    let remainder = size % buffer_alignment;
    if remainder != 0 {
        buffer_alignment - remainder
    } else {
        0
    }
}

impl Slice {
    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    pub(crate) fn size(&self) -> usize {
        self.effective_size()
    }

    pub(crate) fn split(&mut self, offset_slice: usize, buffer_alignment: usize) -> Option<Self> {
        let size_new = self.effective_size() - offset_slice;
        let offset_new = self.storage.offset() + offset_slice;
        let old_size = self.effective_size();

        let storage_new = StorageHandle {
            id: self.storage.id,
            utilization: StorageUtilization::Slice {
                offset: offset_new,
                size: size_new,
            },
        };

        self.storage.utilization = StorageUtilization::Slice {
            offset: self.storage.offset(),
            size: offset_slice,
        };

        if offset_new > 0 && size_new < MIN_SIZE_NEEDED_TO_OFFSET {
            panic!("tried to create slice of size {size_new} with an offset while the size needs to atleast be of size {MIN_SIZE_NEEDED_TO_OFFSET} for offset support");
        }
        if offset_new % buffer_alignment != 0 {
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

    pub(crate) fn id(&self) -> SliceId {
        *self.handle.id()
    }

    pub(crate) fn next_slice_position(&self) -> usize {
        self.storage.offset() + self.effective_size()
    }
}

impl Chunk {
    pub(crate) fn merge_next_slice(
        &mut self,
        from_slice_index: usize,
        slices: &mut HashMap<SliceId, Slice>,
    ) -> bool {
        self.slices.merge_with_next_slice(from_slice_index, slices)
    }

    pub(crate) fn slice(&self, index: usize) -> Option<SliceId> {
        self.slices.find_slice(index)
    }

    pub(crate) fn insert_slice(
        &mut self,
        position: usize,
        slice: Slice,
        slices: &mut HashMap<SliceId, Slice>,
    ) {
        self.slices.insert_slice(position, slice.id());
        slices.insert(slice.id(), slice);
    }
}
