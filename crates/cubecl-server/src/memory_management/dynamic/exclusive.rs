//! One page per allocation, in size buckets.

use crate::memory_management::Cleanup;
use crate::{
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryPoolReport, MemoryUsage,
        memory_pool::{ExclusiveMemoryPool, MemoryPool, PageMapping},
    },
    server::IoError,
    storage::ComputeStorage,
};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::MemoryDeviceProperties;

/// The smallest bucket, in bytes.
const MIN_BUCKET_SIZE: u64 = 1024 * 32;
/// How many buckets span the sizes up to the device's largest page, before
/// alignment folds some of them together.
const MAX_BUCKETS: usize = 24;
/// The allocations a page waits through, unused, before it is released.
const BASE_DEALLOC_PERIOD: u64 = 5000;
/// Bytes of page size that add another [`BASE_DEALLOC_PERIOD`] to the wait.
const DEALLOC_SCALE: u64 = 1024 * 1024 * 1024;

/// One page per allocation, in exponentially spaced size buckets: what a
/// device that cannot sub-slice gets, and what a staging or uniform pool
/// wants whatever the device. Nothing is ever outdated here.
///
/// A bucket is addressed by the index a slice's location carries.
pub struct ExclusivePools {
    buckets: Vec<ExclusiveMemoryPool>,
}

impl ExclusivePools {
    /// The buckets for a device with `properties`.
    pub fn new(properties: &MemoryDeviceProperties) -> Self {
        let buckets = bucket_sizes(properties.max_page_size, properties.alignment)
            .into_iter()
            .enumerate()
            .map(|(index, size)| {
                // Larger pages wait longer before going back: they cost more to
                // allocate again.
                let dealloc_period = (BASE_DEALLOC_PERIOD as f64
                    * (1.0 + size as f64 / DEALLOC_SCALE as f64).round())
                    as u64;
                ExclusiveMemoryPool::new(size, properties.alignment, dealloc_period, index as u8)
            })
            .collect();
        Self { buckets }
    }

    /// The bucket `index` names.
    pub fn pool(&self, index: u8) -> Option<&ExclusiveMemoryPool> {
        self.buckets.get(index as usize)
    }

    /// The bucket `index` names, mutably.
    pub fn pool_mut(&mut self, index: u8) -> Option<&mut ExclusiveMemoryPool> {
        self.buckets.get_mut(index as usize)
    }

    /// Reserve `size` bytes on a page a bucket already holds, else allocate one
    /// on the smallest bucket that accepts the size.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no bucket accepts the size, and whatever
    /// the device refused when one does.
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        if let Some(handle) = self.try_reserve(size, failures) {
            return Ok(handle);
        }
        match self.buckets.iter_mut().find(|pool| pool.accept(size)) {
            Some(pool) => pool.alloc(storage, size, mapping, failures),
            None => Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// Reserve `size` bytes on a page a bucket already holds. `None` when
    /// none has one free.
    pub fn try_reserve(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        self.buckets
            .iter_mut()
            .filter(|pool| pool.accept(size))
            .find_map(|pool| pool.try_reserve(size, failures))
    }

    /// Release what the buckets no longer need.
    pub fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        for pool in self.buckets.iter_mut() {
            pool.cleanup(storage, alloc_nr, cleanup, failures);
        }
    }

    /// The usage of every bucket.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.buckets
            .iter()
            .fold(MemoryUsage::default(), |usage, pool| {
                usage.combine(pool.get_memory_usage())
            })
    }

    /// A report per bucket, smallest first.
    pub fn report(&self) -> Vec<MemoryPoolReport> {
        self.buckets
            .iter()
            .map(ExclusiveMemoryPool::report)
            .collect()
    }
}

impl core::fmt::Display for ExclusivePools {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for pool in self.buckets.iter() {
            write!(f, "{pool}")?;
        }
        Ok(())
    }
}

/// Exponentially spaced bucket sizes from [`MIN_BUCKET_SIZE`] to `max_size`,
/// aligned, the ones alignment folded together counted once.
fn bucket_sizes(max_size: u64, alignment: u64) -> Vec<u64> {
    let log_min = (MIN_BUCKET_SIZE as f64).ln();
    let log_range = (max_size as f64).ln() - log_min;

    let mut sizes: Vec<u64> = (0..MAX_BUCKETS)
        .map(|index| {
            let position = index as f64 / (MAX_BUCKETS - 1) as f64;
            let size = (log_min + log_range * position).exp() as u64;
            size.next_multiple_of(alignment)
        })
        .collect();
    sizes.dedup();
    sizes
}
