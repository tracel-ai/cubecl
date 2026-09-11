use crate::compute::threadpool::completion_counter::CompletionCounter;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{
    BytesResource, BytesStorage, ComputeStorage, StorageHandle, StorageId,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Byte storage that holds a deallocation back until the stream has executed
/// every unit that was queued when the memory was released, so a launch still
/// waiting in the pool never reads a freed page.
///
/// Entries still behind their fence are left allocated at teardown, as
/// [`BytesStorage`] leaves its own map, which is what keeps those pointers
/// valid for whatever the pool never got to run.
pub struct OrderedStorage {
    inner: BytesStorage,
    pending: VecDeque<(u64, StorageId)>,
    progress: Arc<CompletionCounter>,
    /// Units the stream has enqueued so far. The client thread is its only
    /// writer and it orders nothing on its own, so every access is relaxed.
    frontier: Arc<AtomicU64>,
}

impl OrderedStorage {
    /// Wraps a fresh [`BytesStorage`] behind the progress of one stream.
    pub fn new(progress: Arc<CompletionCounter>, frontier: Arc<AtomicU64>) -> Self {
        Self {
            inner: BytesStorage::default(),
            pending: VecDeque::new(),
            progress,
            frontier,
        }
    }

    /// Frees what the stream has run past. The frontier only grows, so the
    /// queue is fence-ordered and stops at the first entry still ahead.
    fn release_reached(&mut self) {
        let progress = self.progress.load();
        while let Some(&(fence, id)) = self.pending.front() {
            if fence > progress {
                break;
            }
            self.pending.pop_front();
            self.inner.dealloc(id);
        }
    }
}

impl ComputeStorage for OrderedStorage {
    type Resource = BytesResource;

    fn alignment(&self) -> usize {
        self.inner.alignment()
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        self.release_reached();
        self.inner.get(handle)
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        self.release_reached();
        self.inner.alloc(size)
    }

    fn dealloc(&mut self, id: StorageId) {
        self.release_reached();
        let fence = self.frontier.load(Ordering::Relaxed);
        if self.progress.load() >= fence {
            self.inner.dealloc(id);
        } else {
            self.pending.push_back((fence, id));
        }
    }

    fn flush(&mut self) {
        self.release_reached();
        self.inner.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Units the stream is pretending to have queued behind the allocation.
    const QUEUED_UNITS: u64 = 4;
    const PAGE_SIZE: u64 = 64;

    fn storage() -> (OrderedStorage, Arc<CompletionCounter>, Arc<AtomicU64>) {
        let progress = Arc::new(CompletionCounter::new());
        let frontier = Arc::new(AtomicU64::new(0));
        (
            OrderedStorage::new(progress.clone(), frontier.clone()),
            progress,
            frontier,
        )
    }

    /// Memory released while units are still queued stays mapped until they run.
    #[test]
    fn a_dealloc_behind_the_frontier_waits_for_the_queued_units() {
        let (mut storage, progress, frontier) = storage();
        let handle = storage.alloc(PAGE_SIZE).unwrap();
        frontier.store(QUEUED_UNITS, Ordering::Relaxed);

        storage.dealloc(handle.id);
        assert!(storage.get(&handle).is_ok());

        for _ in 0..QUEUED_UNITS {
            progress.add_done();
        }
        assert!(storage.get(&handle).is_err());
    }

    /// Memory released with nothing left to run is freed on the spot.
    #[test]
    fn a_dealloc_at_the_frontier_frees_immediately() {
        let (mut storage, _progress, _frontier) = storage();
        let handle = storage.alloc(PAGE_SIZE).unwrap();

        storage.dealloc(handle.id);
        assert!(storage.pending.is_empty());
        assert!(storage.get(&handle).is_err());
    }
}
