//! One dynamic pool, whichever kind it is.

use crate::{
    memory_management::{
        ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryPoolOptions, MemoryPoolReport,
        MemoryUsage, PoolType,
        memory_pool::{ExclusiveMemoryPool, MemoryPool, PageMapping, Slice, SlicedPool},
    },
    server::IoError,
    storage::ComputeStorage,
};
use cubecl_ir::MemoryDeviceProperties;

// These are 288 bytes vs 64 bytes. Adding boxing isn't really worth
// saving the 200 bytes.
#[allow(clippy::large_enum_variant)]
pub enum DynamicPool {
    Sliced(SlicedPool),
    Exclusive(ExclusiveMemoryPool),
}

impl MemoryPool for DynamicPool {
    fn accept(&self, size: u64) -> bool {
        match self {
            DynamicPool::Sliced(pool) => pool.accept(size),
            DynamicPool::Exclusive(pool) => pool.accept(size),
        }
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.find(binding),
            DynamicPool::Exclusive(m) => m.find(binding),
        }
    }

    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.find_mut(binding),
            DynamicPool::Exclusive(m) => m.find_mut(binding),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
    fn try_reserve(&mut self, size: u64, failures: &mut ErrorGraph) -> Option<ManagedMemoryHandle> {
        match self {
            DynamicPool::Sliced(m) => m.try_reserve(size, failures),
            DynamicPool::Exclusive(m) => m.try_reserve(size, failures),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.alloc(storage, size, mapping, failures),
            DynamicPool::Exclusive(m) => m.alloc(storage, size, mapping, failures),
        }
    }

    fn materialize<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        match self {
            DynamicPool::Sliced(m) => m.materialize(storage, binding),
            DynamicPool::Exclusive(m) => m.materialize(storage, binding),
        }
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicPool::Sliced(m) => m.get_memory_usage(),
            DynamicPool::Exclusive(m) => m.get_memory_usage(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
        failures: &mut ErrorGraph,
    ) {
        match self {
            DynamicPool::Sliced(m) => m.cleanup(storage, alloc_nr, explicit, failures),
            DynamicPool::Exclusive(m) => m.cleanup(storage, alloc_nr, explicit, failures),
        };
        storage.flush();
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        match self {
            DynamicPool::Sliced(m) => m.bind(reserved, assigned, cursor, failures),
            DynamicPool::Exclusive(m) => m.bind(reserved, assigned, cursor, failures),
        }
    }
}

impl core::fmt::Display for DynamicPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DynamicPool::Sliced(pool) => write!(f, "{pool}"),
            DynamicPool::Exclusive(pool) => write!(f, "{pool}"),
        }
    }
}

impl DynamicPool {
    pub(crate) fn report(&self) -> MemoryPoolReport {
        match self {
            DynamicPool::Sliced(m) => m.report(m.kind()),
            DynamicPool::Exclusive(m) => m.report(),
        }
    }
}

impl DynamicPool {
    /// The pool `options` asks for, at `pool_pos`.
    ///
    /// `up_to_max_slice` keeps a sliced pool from also taking the allocations
    /// that fill most of a page: they belong to the pool behind it, whose
    /// pages are sized to what they serve.
    pub fn new(
        options: &MemoryPoolOptions,
        properties: &MemoryDeviceProperties,
        pool_pos: u8,
        up_to_max_slice: bool,
    ) -> Self {
        match options.pool_type {
            PoolType::SlicedPages {
                page_size,
                max_slice_size,
            } => {
                let pool =
                    SlicedPool::new(page_size, max_slice_size, properties.alignment, pool_pos);
                DynamicPool::Sliced(match up_to_max_slice {
                    true => pool.up_to_max_slice(),
                    false => pool,
                })
            }
            PoolType::AdaptivePages { min_page_size } => DynamicPool::Sliced(SlicedPool::new(
                min_page_size,
                min_page_size,
                properties.alignment,
                pool_pos,
            )),
            PoolType::ExclusivePages { max_alloc_size } => {
                DynamicPool::Exclusive(ExclusiveMemoryPool::new(
                    max_alloc_size,
                    properties.alignment,
                    options.dealloc_period.unwrap_or(u64::MAX),
                    pool_pos,
                ))
            }
        }
    }
}
