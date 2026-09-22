//! How a stream's dynamic pools are managed.

use super::{AdaptiveMemory, ExclusivePools};
use crate::memory_management::Cleanup;
use crate::memory_management::relocation::{CopyQueue, RelocationNeed, RelocationReason};
use crate::{
    config::memory::MemoryLogLevel,
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, ManagedMemoryBinding, ManagedMemoryHandle, MemoryConfiguration,
        MemoryPoolReport,
        memory_pool::{MemoryPool, PageMapping},
    },
    server::IoError,
    storage::ComputeStorage,
};
use alloc::{boxed::Box, format, string::String, vec::Vec};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

/// The dynamic pools a stream allocates from, and which of them serves what.
///
/// Everything a workload allocates and frees as it runs, as against the
/// persistent memory its weights sit in and the dedicated buffers that own
/// their allocation.
pub enum DynamicMemory {
    /// One page per allocation, in size buckets. Nothing is ever outdated.
    Exclusive(ExclusivePools),
    /// Pages sized to the largest allocation served, with a pool per size:
    /// the pool a growth leaves behind drains and is dropped. Boxed: it holds
    /// its pools inline, and a memory is built once.
    Adaptive(Box<AdaptiveMemory>),
}

impl DynamicMemory {
    /// The pools `config` lays out on a device with `properties`.
    pub fn new(
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        match config {
            MemoryConfiguration::ExclusivePages => {
                let pools = ExclusivePools::new(properties);
                logger.log_memory(
                    |level| !matches!(level, MemoryLogLevel::Disabled),
                    || format!("[{name}] Using memory pools:\n{pools}"),
                );
                DynamicMemory::Exclusive(pools)
            }
            // `Adaptive`, which only exists where `cubecl-runtime` allows
            // sub-slicing. That crate alone decides, so this one cannot name
            // the variant without disagreeing with it once the feature is
            // turned on there directly.
            #[allow(unreachable_patterns)]
            _ => DynamicMemory::Adaptive(Box::new(AdaptiveMemory::new(properties, logger, name))),
        }
    }

    /// The pool `index` names, while one is there.
    pub fn pool(&self, index: u8) -> Option<&dyn MemoryPool> {
        match self {
            DynamicMemory::Exclusive(pools) => Some(pools.pool(index)?),
            DynamicMemory::Adaptive(memory) => memory.pool(index),
        }
    }

    /// The pool `index` names, mutably.
    pub fn pool_mut(&mut self, index: u8) -> Option<&mut dyn MemoryPool> {
        match self {
            DynamicMemory::Exclusive(pools) => Some(pools.pool_mut(index)?),
            DynamicMemory::Adaptive(memory) => memory.pool_mut(index),
        }
    }

    /// Install real backing behind `binding` when its allocation was carved
    /// lazily. Only the adaptive pages are ever lazy.
    pub fn materialize<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        match self {
            DynamicMemory::Exclusive(_) => Ok(()),
            DynamicMemory::Adaptive(memory) => memory.materialize(storage, binding),
        }
    }

    /// Reserve `size` bytes on the pool that serves them.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no pool accepts the size, and whatever
    /// the device refused when one does.
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match self {
            DynamicMemory::Exclusive(pools) => pools.reserve(storage, size, mapping, failures),
            DynamicMemory::Adaptive(memory) => memory.reserve(storage, size, mapping, failures),
        }
    }

    /// Reserve `size` bytes in the room a pool already holds, without growing
    /// anything. `None` when none has room for it.
    pub fn try_reserve(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        match self {
            DynamicMemory::Exclusive(pools) => pools.try_reserve(size, failures),
            DynamicMemory::Adaptive(memory) => memory.try_reserve(size, failures),
        }
    }

    /// Whether a relocation is wanted, before the bytes the device holds are
    /// known. Never where pages are never outdated.
    pub fn relocation_need(&self) -> RelocationNeed {
        match self {
            DynamicMemory::Exclusive(_) => RelocationNeed::Nothing,
            DynamicMemory::Adaptive(memory) => memory.relocation_need(),
        }
    }

    /// Empty what the outdated pools hold into the current pages, so the
    /// pages they held go back to the driver. Nothing to move where pages are
    /// never outdated.
    pub fn relocate<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        copier: &mut dyn CopyQueue<Storage>,
        reason: RelocationReason,
        failures: &mut ErrorGraph,
    ) {
        match self {
            DynamicMemory::Exclusive(_) => {}
            DynamicMemory::Adaptive(memory) => memory.relocate(storage, copier, reason, failures),
        }
    }

    /// Release what the pools no longer need.
    pub fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        match self {
            DynamicMemory::Exclusive(pools) => pools.cleanup(storage, alloc_nr, cleanup, failures),
            DynamicMemory::Adaptive(memory) => memory.cleanup(storage, alloc_nr, cleanup, failures),
        }
    }

    /// A report per pool held, in the order allocations are routed through
    /// them.
    pub fn report(&self) -> Vec<MemoryPoolReport> {
        match self {
            DynamicMemory::Exclusive(pools) => pools.report(),
            DynamicMemory::Adaptive(memory) => memory.report(),
        }
    }
}

impl core::fmt::Display for DynamicMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DynamicMemory::Exclusive(pools) => write!(f, "{pools}"),
            DynamicMemory::Adaptive(memory) => write!(f, "{memory}"),
        }
    }
}
