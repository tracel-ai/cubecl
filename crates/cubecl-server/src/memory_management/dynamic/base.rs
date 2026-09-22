//! How a stream's dynamic pools are managed.

use super::{AdaptiveMemory, DynamicPool, Pools};
use crate::memory_management::relocation::Move;
use crate::{
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryConfiguration, MemoryPoolReport, MemoryUsage,
        PoolType, memory_pool::PageMapping,
    },
    server::IoError,
    storage::ComputeStorage,
};
use alloc::{string::String, vec::Vec};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

/// The dynamic pools a stream allocates from, and which of them serves what.
///
/// Everything a workload allocates and frees as it runs, as against the
/// persistent memory its weights sit in and the dedicated buffers that own
/// their allocation.
pub enum DynamicMemory {
    /// One page per allocation, in exponentially spaced size buckets: what a
    /// device that cannot sub-slice gets, and what a staging or uniform pool
    /// wants whatever the device. Nothing is ever outdated here.
    Exclusive(Pools),
    /// Pages sized to the largest allocation served, with a pool per size:
    /// the pool a growth leaves behind drains and is dropped.
    Adaptive(AdaptiveMemory),
}

impl DynamicMemory {
    /// The pools `config` asks for, on a device with `properties`.
    pub fn new(
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        let options = config.pool_options(properties);
        let adaptive = options
            .iter()
            .any(|pool| matches!(pool.pool_type, PoolType::AdaptivePages { .. }));

        match adaptive {
            true => DynamicMemory::Adaptive(AdaptiveMemory::new(properties, options, logger, name)),
            false => {
                DynamicMemory::Exclusive(Pools::new(properties, &options, false, logger, name))
            }
        }
    }

    /// The pool `index` names, while one is there.
    pub fn get(&self, index: usize) -> Option<&DynamicPool> {
        match self {
            DynamicMemory::Exclusive(pools) => pools.get(index),
            DynamicMemory::Adaptive(memory) => memory.get(index),
        }
    }

    /// The pool `index` names, mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DynamicPool> {
        match self {
            DynamicMemory::Exclusive(pools) => pools.get_mut(index),
            DynamicMemory::Adaptive(memory) => memory.get_mut(index),
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
            DynamicMemory::Exclusive(pools) => {
                let routing = 0..pools.len() as u8;
                pools.reserve(routing, storage, size, mapping, failures)
            }
            DynamicMemory::Adaptive(memory) => memory.reserve(storage, size, mapping, failures),
        }
    }

    /// Release what the pools no longer need.
    pub fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
        failures: &mut ErrorGraph,
    ) {
        match self {
            DynamicMemory::Exclusive(pools) => pools.cleanup(storage, alloc_nr, explicit, failures),
            DynamicMemory::Adaptive(memory) => {
                memory.cleanup(storage, alloc_nr, explicit, failures)
            }
        }
    }

    /// The usage of every pool held.
    pub fn memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicMemory::Exclusive(pools) => pools.memory_usage(),
            DynamicMemory::Adaptive(memory) => memory.memory_usage(),
        }
    }

    /// A report per pool held, in the order allocations are routed through
    /// them.
    pub fn report(&self) -> Vec<MemoryPoolReport> {
        match self {
            DynamicMemory::Exclusive(pools) => pools.report(0..pools.len() as u8),
            DynamicMemory::Adaptive(memory) => memory.report(),
        }
    }

    /// Reserve a target for every allocation left on an outdated pool's pages
    /// (see [`AdaptiveMemory::plan_relocation`]). Nothing to move where pages
    /// are never outdated.
    pub fn plan_relocation<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        failures: &mut ErrorGraph,
    ) -> Vec<Move> {
        match self {
            DynamicMemory::Exclusive(_) => Vec::new(),
            DynamicMemory::Adaptive(memory) => memory.plan_relocation(storage, failures),
        }
    }

    /// Hand a planned allocation over to its target, once its bytes are there.
    pub fn commit_relocation(&mut self, relocated: Move, failures: &mut ErrorGraph) {
        match self {
            DynamicMemory::Exclusive(_) => {}
            DynamicMemory::Adaptive(memory) => memory.commit_relocation(relocated, failures),
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
