use crate::{
    execution::{ProcessMode, StreamMode},
    server::IoError,
    storage::{StorageAllocator, StorageHandle, StorageId, StorageUtilization},
};

/// Whether a fresh device page gets real backing at allocation time.
///
/// Either way the pool's bookkeeping — slice carving, coalescing, high-water
/// marks — is identical; the two differ only in when the driver is asked for
/// memory. [`PageMapping::current`] decides which an allocation gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageMapping {
    /// Allocate device memory now.
    ///
    /// The answer whenever the allocation will be used, which unless launches
    /// are discarded is all of them: a reservation becomes a kernel argument, a
    /// read or a write within microseconds, so deferring it buys nothing — and
    /// gives up the reservation-time failure the backends can still recover
    /// (`Command::reserve` retries after reclaiming the stream), the pure
    /// lookup that keeps resolution infallible on the launch paths, and the
    /// guarantee that a capture window never allocates once recording starts.
    Eager,
    /// Mint the storage id now and defer the device allocation to the first
    /// time one of the page's slices is resolved
    /// ([`SliceReserver::materialize`](super::SliceReserver::materialize)).
    /// Until then the id has no driver memory behind it and the page's device
    /// footprint is zero.
    ///
    /// For the one case where the allocation may never be used: under a
    /// [process mode](ProcessMode) that discards launches, the
    /// workload's launches are compiled and discarded, so most reservations are
    /// never resolved and never need to exist. That is what lets a workload far
    /// larger than the device replay its allocation stream — the pools still
    /// measure it, and only what genuinely executes (a tuning pass, which
    /// resolves because it runs) costs real memory.
    Lazy,
}

impl From<ProcessMode> for PageMapping {
    /// [`Lazy`](Self::Lazy) under a process mode whose streams discard their
    /// launches, [`Eager`](Self::Eager) otherwise: a stream that executes
    /// while the process mode discards is a measurement, which resolves what it
    /// uses because it runs.
    fn from(mode: ProcessMode) -> Self {
        match mode.stream_mode() {
            StreamMode::Discard => Self::Lazy,
            StreamMode::Execute => Self::Eager,
        }
    }
}

impl PageMapping {
    /// The mapping allocations made in this process, right now, get: what the
    /// [process mode](ProcessMode) in force asks.
    pub fn current() -> Self {
        Self::from(ProcessMode::current())
    }
}

/// A storage handle for `size` bytes honoring `mapping`: a real device
/// allocation when [`Eager`](PageMapping::Eager), a minted id awaiting
/// [`SliceReserver::materialize`](super::SliceReserver::materialize) when
/// [`Lazy`](PageMapping::Lazy).
///
/// # Errors
///
/// Whatever the storage refuses, for an eager mapping.
pub(crate) fn page_storage<Storage: StorageAllocator>(
    mapping: PageMapping,
    storage: &mut Storage,
    size: u64,
) -> Result<StorageHandle, IoError> {
    match mapping {
        PageMapping::Eager => storage.allocate_bytes(size),
        PageMapping::Lazy => Ok(StorageHandle::new(
            StorageId::new(),
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: size,
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BytesStorage;

    /// A lazy page mints an id the storage never allocated, and an eager one
    /// is allocated now.
    #[test]
    fn only_an_eager_page_is_allocated_now() {
        let mut storage = BytesStorage::default();

        let lazy = page_storage(PageMapping::Lazy, &mut storage, 64)
            .expect("a minted id should never fail");
        assert_eq!((lazy.size(), storage.bytes_allocated()), (64, 0));

        page_storage(PageMapping::Eager, &mut storage, 64).expect("the host should have 64 bytes");
        assert_eq!(storage.bytes_allocated(), 64);
    }
}
