//! One page per allocation, in size buckets.

use crate::memory_management::Cleanup;
use crate::{
    memory_management::{
        ErrorGraph, ManagedMemoryHandle, MemoryPoolReport,
        memory_pool::{ExclusiveLayout, ExclusiveSliceReserver, PageMapping, SliceReserver},
    },
    server::IoError,
    storage::StorageAllocator,
};
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::MemoryDeviceProperties;

/// The smallest bucket.
const MIN_BUCKET_BYTES: u64 = 1024 * 32;
/// How many buckets span the sizes up to the device's largest page, before
/// alignment folds some of them together.
const MAX_BUCKET_COUNT: usize = 24;
/// The reservations a page waits through, unused, before it is released.
const BASE_DEALLOC_PERIOD_COUNT: u64 = 5000;
/// Bytes of page size that add another [`BASE_DEALLOC_PERIOD_COUNT`] to the wait.
const DEALLOC_SCALE_BYTES: u64 = 1024 * 1024 * 1024;

/// Reserves one page per allocation, in exponentially spaced size buckets:
/// what a device that cannot sub-slice gets, and what a staging or uniform
/// pool wants whatever the device. Nothing is ever outdated here.
///
/// A bucket is addressed by the index a slice's location carries.
pub(crate) struct BucketReserver {
    /// The buckets, smallest first, read by the index a location carries
    /// (Position) and scanned in size order for the first that accepts a
    /// size (Ordered).
    buckets: Vec<ExclusiveSliceReserver>,
}

impl BucketReserver {
    /// The buckets for a device with `properties`.
    pub(crate) fn new(properties: &MemoryDeviceProperties) -> Self {
        let buckets = bucket_sizes(properties.max_page_size, properties.alignment)
            .into_iter()
            .enumerate()
            .map(|(index, size)| {
                ExclusiveSliceReserver::new(ExclusiveLayout {
                    max_alloc_bytes: size,
                    alignment_bytes: properties.alignment,
                    dealloc_period_count: dealloc_period_for_bytes(size),
                    pool: index as u8,
                })
            })
            .collect();
        Self { buckets }
    }

    /// The bucket `pool_index` names.
    pub(crate) fn pool_by_index(&self, pool_index: u8) -> Option<&ExclusiveSliceReserver> {
        self.buckets.get(pool_index as usize)
    }

    /// The bucket `pool_index` names, mutably.
    pub(crate) fn pool_by_index_mut(
        &mut self,
        pool_index: u8,
    ) -> Option<&mut ExclusiveSliceReserver> {
        self.buckets.get_mut(pool_index as usize)
    }

    /// Reserve `size` bytes on a page a bucket already holds, else allocate one
    /// on the smallest bucket that accepts the size.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no bucket accepts the size, and whatever
    /// the device refused when one does.
    pub(crate) fn reserve<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        if let Some(handle) = self.try_reserve(size, failures) {
            return Ok(handle);
        }
        let Some(bucket) = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.accepts_bytes(size))
        else {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        };
        bucket.alloc(storage, size, mapping, failures)
    }

    /// Reserve `size` bytes on a page a bucket already holds. `None` when
    /// none has one free.
    pub(crate) fn try_reserve(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        self.buckets
            .iter_mut()
            .filter(|bucket| bucket.accepts_bytes(size))
            .find_map(|bucket| bucket.try_reserve(size, failures))
    }

    /// Release what the buckets no longer need.
    pub(crate) fn cleanup<Storage: StorageAllocator>(
        &mut self,
        storage: &mut Storage,
        reservation_count: u64,
        cleanup: Cleanup,
        failures: &mut ErrorGraph,
    ) {
        for bucket in self.buckets.iter_mut() {
            bucket.cleanup(storage, reservation_count, cleanup, failures);
        }
    }

    /// A report per bucket, smallest first.
    pub(crate) fn report(&self) -> Vec<MemoryPoolReport> {
        self.buckets
            .iter()
            .map(ExclusiveSliceReserver::report)
            .collect()
    }
}

impl core::fmt::Display for BucketReserver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for bucket in self.buckets.iter() {
            write!(f, "{bucket}")?;
        }
        Ok(())
    }
}

/// The reservations a page of `size` bytes waits through, unused, before it
/// goes back: larger pages wait longer, since they cost more to allocate
/// again.
fn dealloc_period_for_bytes(size: u64) -> u64 {
    (BASE_DEALLOC_PERIOD_COUNT as f64 * (1.0 + size as f64 / DEALLOC_SCALE_BYTES as f64).round())
        as u64
}

/// Exponentially spaced bucket sizes from [`MIN_BUCKET_BYTES`] to `max_size`,
/// aligned, the ones alignment folded together counted once.
fn bucket_sizes(max_size: u64, alignment: u64) -> Vec<u64> {
    let log_min = (MIN_BUCKET_BYTES as f64).ln();
    let log_range = (max_size as f64).ln() - log_min;

    let mut sizes: Vec<u64> = (0..MAX_BUCKET_COUNT)
        .map(|index| {
            let position = index as f64 / (MAX_BUCKET_COUNT - 1) as f64;
            let size = (log_min + log_range * position).exp() as u64;
            size.next_multiple_of(alignment)
        })
        .collect();
    sizes.dedup();
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BytesStorage;

    const MIB: u64 = 1024 * 1024;

    fn buckets() -> BucketReserver {
        BucketReserver::new(&MemoryDeviceProperties::new(16 * MIB, 32))
    }

    /// The buckets span the smallest bucket to the largest page, smallest first.
    #[test_log::test]
    fn the_buckets_span_the_smallest_to_the_largest_page() {
        let sizes = bucket_sizes(16 * MIB, 32);

        assert_eq!(sizes.first(), Some(&MIN_BUCKET_BYTES));
        assert_eq!(sizes.last(), Some(&(16 * MIB)));
        assert!(sizes.is_sorted());
    }

    /// A size lands on the smallest bucket that accepts it.
    #[test_log::test]
    fn a_size_lands_on_the_smallest_bucket_that_accepts_it() {
        let mut storage = BytesStorage::default();
        let mut buckets = buckets();

        let handle = buckets
            .reserve(
                &mut storage,
                MIN_BUCKET_BYTES + 1,
                PageMapping::Eager,
                &mut ErrorGraph::default(),
            )
            .expect("a bucket should accept the size");

        assert_eq!(handle.descriptor().location().pool, 1);
    }

    /// A size no bucket accepts is refused.
    #[test_log::test]
    fn a_size_no_bucket_accepts_is_refused() {
        let mut storage = BytesStorage::default();
        let mut buckets = buckets();

        let refused = buckets.reserve(
            &mut storage,
            16 * MIB + 1,
            PageMapping::Eager,
            &mut ErrorGraph::default(),
        );

        assert!(matches!(refused, Err(IoError::BufferTooBig { .. })));
    }

    /// A freed page is reused before a new one is allocated.
    #[test_log::test]
    fn a_freed_page_is_reused() {
        let mut storage = BytesStorage::default();
        let mut buckets = buckets();
        let first = buckets
            .reserve(
                &mut storage,
                1024,
                PageMapping::Eager,
                &mut ErrorGraph::default(),
            )
            .expect("a bucket should accept the size");
        core::mem::drop(first);

        assert!(
            buckets
                .try_reserve(1024, &mut ErrorGraph::default())
                .is_some()
        );
        assert_eq!(
            buckets
                .report()
                .iter()
                .map(|report| report.page_count)
                .sum::<u64>(),
            1
        );
    }
}
