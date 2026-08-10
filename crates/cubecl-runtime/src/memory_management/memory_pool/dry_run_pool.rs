use super::{ManagedMemoryBinding, ManagedMemoryHandle, MemoryPool, PersistentPool, Slice};
use crate::memory_management::{MemoryPoolReport, MemoryUsage};
use crate::server::IoError;
use crate::storage::ComputeStorage;

/// The pool serving the allocations measurements make inside a dry run.
///
/// A dry run exists to measure a workload: its launches are compiled and
/// dropped, its allocation stream still runs, and the dynamic pools'
/// high-water marks become the workload's memory plan
/// ([`MemoryReport`](crate::memory_management::MemoryReport)). The exception
/// inside that stream is autotune: its benchmarks execute for real
/// ([`RealRun`](crate::dry_run::RealRun)), and what they allocate is scratch
/// for the measurement, not part of the workload — served from the dynamic
/// pools it would inflate the very peaks being measured. This pool isolates
/// it; [`serves`](Self::serves) is the routing predicate.
///
/// An exact-fit [`PersistentPool`] underneath: benchmark iterations repeat
/// the same shapes, so exact-size reuse serves them with zero padding, and
/// the slices stay until an explicit cleanup. Like the persistent pool, it
/// lives at a fixed sentinel position and survives dynamic-pool rebuilds —
/// a plan is read and re-installed right after the dry run that filled this
/// pool, and rebuilding must not disturb it. Empty outside dry runs, so it
/// costs nothing when no plan is being measured.
pub(crate) struct DryRunPool {
    pool: PersistentPool,
}

impl DryRunPool {
    /// Create the pool, stamped at `pool_pos` (the caller's sentinel).
    /// Accepts any size: what a measurement allocates is not the caller's to
    /// bound, and the storage errors on a truly unallocatable size anyway.
    pub fn new(alignment: u64, pool_pos: u8) -> Self {
        Self {
            pool: PersistentPool::new(u64::MAX, alignment, pool_pos),
        }
    }

    /// Whether an allocation being made right now belongs here: a dry run is
    /// active and this thread is issuing a measurement. Meaningful on the
    /// thread performing the allocation, which is the thread that asked for
    /// it — server access is serialized, so allocations are served
    /// synchronously on the issuing thread.
    pub fn serves() -> bool {
        crate::dry_run::dry_run() && crate::dry_run::measuring()
    }

    /// Serve a measurement's allocation: an exact-size free slice when one
    /// exists, a fresh device allocation otherwise.
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        if let Some(handle) = self.pool.try_reserve(size) {
            return Ok(handle);
        }
        self.pool.alloc(storage, size)
    }

    pub fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        self.pool.find(binding)
    }

    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        self.pool.bind(reserved, assigned, cursor)
    }

    pub fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        self.pool.cleanup(storage, alloc_nr, explicit);
    }

    pub fn get_memory_usage(&self) -> MemoryUsage {
        self.pool.get_memory_usage()
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub fn report(&self) -> MemoryPoolReport {
        self.pool.report()
    }
}

impl core::fmt::Display for DryRunPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.pool, f)
    }
}
