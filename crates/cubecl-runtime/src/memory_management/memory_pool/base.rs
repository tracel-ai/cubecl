use std::fmt::Display;

use super::{SliceBinding, SliceHandle, SliceId};
use crate::storage::{ComputeStorage, StorageHandle, StorageId};

#[derive(new, Debug)]
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: usize,
}

impl Slice {
    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    pub(crate) fn effective_size(&self) -> usize {
        self.storage.size() + self.padding
    }

    pub(crate) fn id(&self) -> SliceId {
        *self.handle.id()
    }
}

pub(crate) fn calculate_padding(size: usize, buffer_alignment: usize) -> usize {
    let remainder = size % buffer_alignment;
    if remainder != 0 {
        buffer_alignment - remainder
    } else {
        0
    }
}

#[derive(Default)]
pub struct MemoryUsage {
    pub number_allocs: usize,
    pub bytes_in_use: usize,
    pub bytes_padding: usize,
    pub bytes_reserved: usize,
}

impl MemoryUsage {
    pub fn combine(&self, other: MemoryUsage) -> MemoryUsage {
        MemoryUsage {
            number_allocs: self.number_allocs + other.number_allocs,
            bytes_in_use: self.bytes_in_use + other.bytes_in_use,
            bytes_padding: self.bytes_padding + other.bytes_padding,
            bytes_reserved: self.bytes_reserved + other.bytes_reserved,
        }
    }
}

impl Display for MemoryUsage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // In the future it'd be nice if MemoryUsage also held some stats about say,
        // the 5 biggest allocations, to show when you an OOM.
        let usage_percentage = (self.bytes_in_use as f32 / self.bytes_reserved as f32) * 100.0;
        let padding_percentage = (self.bytes_padding as f32 / self.bytes_in_use as f32) * 100.0;
        writeln!(f, "Memory Usage Report:")?;
        writeln!(f, "  Number of allocations: {}", self.number_allocs)?;
        writeln!(f, "  Bytes in use: {} bytes", self.bytes_in_use)?;
        writeln!(f, "  Bytes used for padding: {} bytes", self.bytes_padding)?;
        writeln!(f, "  Total bytes reserved: {} bytes", self.bytes_reserved)?;
        writeln!(f, "  Usage efficiency: {:.2}%", usage_percentage)?;
        writeln!(f, "  Padding overhead: {:.2}%", padding_percentage)
    }
}

pub trait MemoryPool {
    fn max_alloc_size(&self) -> usize;

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle>;

    fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: usize,
        exclude: &[StorageId],
    ) -> SliceHandle;

    fn alloc<Storage: ComputeStorage>(&mut self, storage: &mut Storage, size: usize)
        -> SliceHandle;

    fn get_memory_usage(&self) -> MemoryUsage;
}

#[derive(Debug, Clone)]
pub enum PoolType {
    ExclusivePages,
    SlicedPages { max_slice_size: usize },
}

/// Options to create a memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolOptions {
    /// What kind of pool to use.
    pub pool_type: PoolType,
    /// The amount of bytes used for each chunk in the memory pool.
    pub page_size: usize,
    /// The number of chunks allocated directly at creation.
    ///
    /// Useful when you know in advance how much memory you'll need.
    pub chunk_num_prealloc: usize,
}
