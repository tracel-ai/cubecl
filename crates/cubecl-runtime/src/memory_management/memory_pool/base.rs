use super::{SliceBinding, SliceHandle, SliceId};
use crate::storage::StorageUtilization;
use crate::storage::VirtualStorage;
use crate::{
    memory_management::MemoryUsage,
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};
use hashbrown::HashMap;

/// This trait is a generalization of the [`Slice`].
/// I created this one and the [`MemoryChunk`] because i needed a generic representation of both,
/// in order to be able to use the [`RingBuffer`] in the [`VirtualMemoryPool`].
pub(crate) trait MemoryFragment {
    /// Any ID built with macro [`memory_id_type!`] can be used in this trait.
    type Key: std::cmp::Eq + std::hash::Hash + Sized + Clone + Copy + PartialEq + std::fmt::Debug;

    /// Whether this memory fragment is free currently free.
    fn is_free(&self) -> bool;

    /// The effective size of the memory fragment (Size + padding)
    fn effective_size(&self) -> u64;

    /// The id of the memory fragment.
    /// Should be obtained by querying the corresponding handle.
    fn id(&self) -> Self::Key;

    /// Utility to split this memory fragment into another, at the specified offset.
    /// If the memory fragment cannot be split, return None by default
    fn split(&mut self, _offset_slice: u64, _buffer_alignment: u64) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// The position of the memory fragment in the chain. Useful for merging and defragmentation algorithms.
    fn next_slice_position(&self) -> u64;
}

/// This other trait is a generalization of a [`MemoryPage`].
/// It represents multiple [`MemoryFragments`] that are expected to be contiguous in the address space.
pub(crate) trait MemoryChunk {
    /// Any ID built with macro [`memory_id_type!`] can be used in this trait.
    type Fragment: MemoryFragment<Key = Self::Key>;
    type Key: std::cmp::Eq + std::hash::Hash + Sized + Clone + Copy + PartialEq + std::fmt::Debug;

    /// merge slice at first_slice_address with the next slice (if there is one and if it's free)
    /// return a boolean representing if a merge happened
    fn merge_with_next_slice(
        &mut self,
        first_slice_address: u64,
        slices: &mut HashMap<Self::Key, Self::Fragment>,
    ) -> bool;

    /// Find an slice at a specific address
    fn find_slice(&self, address: u64) -> Option<Self::Key>;

    /// Insert a slice at a specific address
    fn insert_slice(&mut self, address: u64, slice: Self::Key);
}

#[derive(new, Debug)]
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: u64,
}

impl MemoryFragment for Slice {
    type Key = SliceId;

    fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding
    }

    fn id(&self) -> Self::Key {
        *self.handle.id()
    }

    fn split(&mut self, offset_slice: u64, buffer_alignment: u64) -> Option<Self> {
        let size_new = self.effective_size() - offset_slice;
        let offset_new = self.storage.offset() + offset_slice;
        let old_size = self.effective_size();

        let storage_new = StorageHandle {
            id: self.storage.id,
            utilization: StorageUtilization {
                offset: offset_new,
                size: size_new,
            },
        };

        self.storage.utilization = StorageUtilization {
            offset: self.storage.offset(),
            size: offset_slice,
        };

        if !offset_new.is_multiple_of(buffer_alignment) {
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

    fn next_slice_position(&self) -> u64 {
        self.storage.offset() + self.effective_size()
    }
}

pub(crate) fn calculate_padding(size: u64, buffer_alignment: u64) -> u64 {
    let remainder = size % buffer_alignment;
    if remainder != 0 {
        buffer_alignment - remainder
    } else {
        0
    }
}

pub trait MemoryPool {
    fn max_alloc_size(&self) -> u64;

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle>;

    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle>;

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError>;

    fn get_memory_usage(&self) -> MemoryUsage;

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    );
}
