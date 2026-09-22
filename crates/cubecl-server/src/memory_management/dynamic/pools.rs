//! The pools a strategy holds, addressed by the index a slice's location
//! carries.

use super::DynamicPool;
use crate::{
    config::memory::MemoryLogLevel,
    logging::ServerLogger,
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryPoolOptions, MemoryPoolReport, MemoryUsage,
        PoolType,
        memory_pool::{MemoryPool, PageMapping, SlicedPool},
    },
    server::IoError,
    storage::ComputeStorage,
};
use alloc::{format, string::String, vec::Vec};
use cubecl_environment::{backtrace::BackTrace, sync::Arc};
use cubecl_ir::MemoryDeviceProperties;

/// The pools a strategy holds.
///
/// A pool is addressed by the index a slice's location carries, so one is
/// dropped by leaving its slot empty rather than by moving the others.
pub struct Pools {
    pools: Vec<Option<DynamicPool>>,
    logger: Arc<ServerLogger>,
    name: String,
}

impl Pools {
    /// The pools `options` asks for. `grows` says a pool whose pages are sized
    /// to what they serve is routed last, which keeps the near-page-size
    /// allocations off the fixed pages ahead of it.
    pub fn new(
        properties: &MemoryDeviceProperties,
        options: &[MemoryPoolOptions],
        grows: bool,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        let pools = options
            .iter()
            .enumerate()
            .map(|(pool_pos, pool)| {
                let pool_pos = pool_pos as u8;
                let up_to_max_slice = grows
                    && options[pool_pos as usize + 1..]
                        .iter()
                        .any(|later| matches!(later.pool_type, PoolType::AdaptivePages { .. }));
                Some(DynamicPool::new(
                    pool,
                    properties,
                    pool_pos,
                    up_to_max_slice,
                ))
            })
            .collect();

        let pools = Self {
            pools,
            logger,
            name,
        };
        pools.log_layout();
        pools
    }

    /// How many slots it has, empty ones included.
    pub fn len(&self) -> usize {
        self.pools.len()
    }

    /// The pool `index` names, while one is there.
    pub fn get(&self, index: usize) -> Option<&DynamicPool> {
        self.pools.get(index)?.as_ref()
    }

    /// The pool `index` names, mutably.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DynamicPool> {
        self.pools.get_mut(index)?.as_mut()
    }

    /// The sliced pool `index` names, while one is there.
    pub fn sliced_mut(&mut self, index: u8) -> Option<&mut SlicedPool> {
        match self.pools.get_mut(index as usize)?.as_mut() {
            Some(DynamicPool::Sliced(pool)) => Some(pool),
            _ => None,
        }
    }

    /// The sliced pools `first` and `second` name, which are distinct.
    pub fn sliced_pair(
        &mut self,
        first: u8,
        second: u8,
    ) -> Option<(&mut SlicedPool, &mut SlicedPool)> {
        let [first, second] = self
            .pools
            .get_disjoint_mut([first as usize, second as usize])
            .ok()?;
        match (first.as_mut()?, second.as_mut()?) {
            (DynamicPool::Sliced(first), DynamicPool::Sliced(second)) => Some((first, second)),
            _ => None,
        }
    }

    /// Put `pool` in a slot an emptied pool left, or in a new one, and say
    /// which.
    pub fn insert(&mut self, pool: impl FnOnce(u8) -> DynamicPool) -> u8 {
        let slot = match self.pools.iter().position(Option::is_none) {
            Some(slot) => slot as u8,
            None => {
                self.pools.push(None);
                (self.pools.len() - 1) as u8
            }
        };
        self.pools[slot as usize] = Some(pool(slot));
        self.log_layout();
        slot
    }

    /// Drop the pool `index` names.
    pub fn remove(&mut self, index: u8) {
        self.pools[index as usize] = None;
    }

    /// Release what every pool no longer needs.
    pub fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
        failures: &mut ErrorGraph,
    ) {
        for pool in self.pools.iter_mut().flatten() {
            pool.cleanup(storage, alloc_nr, explicit, failures);
        }
    }

    /// The usage of every pool held.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.pools
            .iter()
            .flatten()
            .fold(MemoryUsage::default(), |usage, pool| {
                usage.combine(pool.get_memory_usage())
            })
    }

    /// A report per pool, in the order `routing` visits them.
    pub fn report(&self, routing: impl Iterator<Item = u8>) -> Vec<MemoryPoolReport> {
        routing
            .filter_map(|index| Some(self.get(index as usize)?.report()))
            .collect()
    }

    fn log_layout(&self) {
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                let mut msg = String::new();
                for pool in self.pools.iter().flatten() {
                    msg += &format!("[{}] Using memory pool: \n {pool}\n", self.name);
                }
                msg
            },
        );
    }

    /// Reserve `size` bytes on the first pool of `routing` that accepts it,
    /// which is what routing an allocation means.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no pool accepts the size, and whatever
    /// the device refused when one does.
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        routing: impl Iterator<Item = u8>,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        for index in routing {
            let Some(pool) = self.pools[index as usize]
                .as_mut()
                .filter(|pool| pool.accept(size))
            else {
                continue;
            };
            if let Some(slice) = pool.try_reserve(size, failures) {
                return Ok(slice);
            }
            return pool.alloc(storage, size, mapping, failures);
        }

        Err(IoError::BufferTooBig {
            size,
            backtrace: BackTrace::capture(),
        })
    }
}

impl core::fmt::Display for Pools {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for pool in self.pools.iter().flatten() {
            f.write_fmt(format_args!("{pool}"))?;
        }
        Ok(())
    }
}
