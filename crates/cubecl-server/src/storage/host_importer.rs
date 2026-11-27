use super::{HostMemoryApi, HostRange, StorageHandle, StorageId, StorageUtilization};
use crate::device_events::{Event, EventApi};
use crate::driver::DriverError;
use crate::server::IoError;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::bytes::Bytes;
use cubecl_environment::collections::HashMap;

/// Imports host memory for one device, pinned through the runtime's
/// [`HostPins`](super::HostPins), and lets go of each import once the work queued before its
/// release is done.
pub struct HostImporter<A: HostMemoryApi> {
    stream: A::Stream,
    /// Keyed by storage id: the device's address of the import, and the memory it holds.
    imported: HashMap<StorageId, (u64, Arc<Bytes>)>,
    /// Released imports the stream may still read. Arrival: in stream order, so they complete in
    /// order and are let go of from the front.
    pending: VecDeque<PendingRelease<A>>,
}

/// Imports released on a stream, let go of once the work queued there before is done. Another
/// stream's reads keep their bindings, so the memory is not released before those complete.
pub struct PendingRelease<A: EventApi> {
    /// Recorded on the stream behind the work that reads `released`.
    pub event: Event<A>,
    /// Arrival: let go of together once the event completes.
    pub released: Vec<Arc<Bytes>>,
}

impl<A: HostMemoryApi> HostImporter<A> {
    /// No imports, released behind the work of `stream`.
    ///
    /// # Safety
    ///
    /// `stream` outlives the importer: every release records an event on it.
    pub unsafe fn on_stream(stream: A::Stream) -> Self {
        Self {
            stream,
            imported: HashMap::new(),
            pending: VecDeque::new(),
        }
    }

    /// Imports `host`, which the device reads until its id is released.
    pub fn import(&mut self, host: &HostRange) -> Result<StorageHandle, IoError> {
        // Imports released since the last flush unpin before this one pins more.
        self.drop_completed();
        let memory = host.memory();
        let refused = |err: DriverError| refused_unless_poisoned(err, memory.len());
        A::host_pins().pin::<A>(memory).map_err(refused)?;
        let first = memory.as_ptr().wrapping_add(host.offset() as usize);
        let ptr = A::host_device_ptr(first)
            .inspect_err(|_| Self::let_go(memory.clone()))
            .map_err(refused)?;
        let id = StorageId::new();
        self.imported.insert(id, (ptr, memory.clone()));
        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: host.len(),
            },
        ))
    }

    /// The device's address of import `id`, or `None` when it is no import.
    pub fn device_ptr(&self, id: &StorageId) -> Option<u64> {
        self.imported.get(id).map(|(ptr, _)| *ptr)
    }

    /// Releases the imports among `ids` behind the work queued on the stream so far.
    pub fn release_after_stream(&mut self, ids: &[StorageId]) {
        let released: Vec<_> = ids
            .iter()
            .filter_map(|id| self.imported.remove(id))
            .map(|(_, memory)| memory)
            .collect();
        self.queue_release(released);
        self.drop_completed();
    }
}

impl<A: HostMemoryApi> HostImporter<A> {
    /// Lets go of `released` once the work queued on the stream so far is done.
    fn queue_release(&mut self, released: Vec<Arc<Bytes>>) {
        if released.is_empty() {
            return;
        }
        match self.record() {
            Ok(event) => self.pending.push_back(PendingRelease { event, released }),
            Err(err) => Self::let_go_unread(released, &err),
        }
    }

    fn record(&self) -> Result<Event<A>, DriverError> {
        let event = Event::new()?;
        event.record(self.stream)?;
        Ok(event)
    }

    /// Lets go of the imports whose release completed.
    fn drop_completed(&mut self) {
        while let Some(release) = self.pending.front() {
            let reached = release.event.is_reached();
            if matches!(reached, Ok(false)) {
                return;
            }
            let Some(PendingRelease { released, .. }) = self.pending.pop_front() else {
                return;
            };
            match reached {
                Ok(_) => released.into_iter().for_each(Self::let_go),
                Err(err) => Self::let_go_unread(released, &err),
            }
        }
    }

    /// Drops this import of `memory`, unpinning it when it was the last; leaked when unpinning
    /// fails, since a device may still read it.
    fn let_go(memory: Arc<Bytes>) {
        if let Err(err) = A::host_pins().unpin::<A>(&memory) {
            leak(memory, &err);
        }
    }

    /// `released`, whose reads `failed` left unknown, let go of where the failure poisoned the
    /// device, whose work stopped with it, and leaked pinned otherwise: a kernel may still read it.
    fn let_go_unread(released: Vec<Arc<Bytes>>, failed: &DriverError) {
        if failed.is_device_poisoned() {
            released.into_iter().for_each(Self::let_go);
            return;
        }
        released.into_iter().for_each(|memory| leak(memory, failed));
    }
}

impl<A: HostMemoryApi> Drop for HostImporter<A> {
    fn drop(&mut self) {
        // The whole device, not this stream: another stream that failed to wait on its reads
        // keeps their bindings, and teardown releases the imports regardless.
        let mut released: Vec<_> = self
            .imported
            .drain()
            .map(|(_, (_, memory))| memory)
            .collect();
        released.extend(self.pending.drain(..).flat_map(|release| release.released));
        if released.is_empty() {
            return;
        }
        match A::device_synchronize() {
            Ok(()) => released.into_iter().for_each(Self::let_go),
            Err(err) => Self::let_go_unread(released, &err),
        }
    }
}

/// Keeps `memory` alive and pinned for good, since the driver may still read it after `err`.
fn leak(memory: Arc<Bytes>, err: &DriverError) {
    log::warn!("leaking imported host memory the driver may still read: {err}");
    core::mem::forget(memory);
}

/// A failed import as [`IoError::HostMemoryRefused`], unless the error poisons the device.
fn refused_unless_poisoned(err: DriverError, size_bytes: usize) -> IoError {
    match err.is_device_poisoned() {
        true => err.into(),
        // A driver's limit on the memory it pins is what refuses a large import.
        false => IoError::HostMemoryRefused {
            reason: format!("importing {size_bytes} bytes of host memory: {err}").into(),
            backtrace: BackTrace::capture(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::host_memory_fake::{devices, host};

    #[test]
    fn devices_importing_one_memory_pin_it_once() {
        let (driver, mut first, mut second) = devices();
        let host = host();
        first.import(&host).unwrap();
        second.import(&host).unwrap();
        assert_eq!(driver.pins.get(), 1);
    }

    #[test]
    fn an_import_spans_its_range() {
        let (_, mut first, _) = devices();
        let handle = first.import(&host()).unwrap();
        assert_eq!(handle.size(), host().len());
    }

    #[test]
    fn the_last_release_unpins() {
        let (driver, mut first, mut second) = devices();
        let host = host();
        let one = first.import(&host).unwrap();
        let other = second.import(&host).unwrap();
        first.release_after_stream(&[one.id]);
        assert_eq!(driver.unpins.get(), 0);
        second.release_after_stream(&[other.id]);
        assert_eq!(driver.unpins.get(), 1);
    }

    #[test]
    fn a_release_waits_for_the_stream() {
        let (driver, mut first, _) = devices();
        let host = host();
        let handle = first.import(&host).unwrap();
        driver.done.set(Some(Ok(false)));
        first.release_after_stream(&[handle.id]);
        assert_eq!(Arc::strong_count(host.memory()), 2);
        driver.done.set(Some(Ok(true)));
        first.release_after_stream(&[]);
        assert_eq!(Arc::strong_count(host.memory()), 1);
        assert_eq!(driver.unpins.get(), 1);
    }

    #[test]
    fn memory_unpinned_by_its_last_release_pins_again() {
        let (driver, mut first, _) = devices();
        let host = host();
        let handle = first.import(&host).unwrap();
        first.release_after_stream(&[handle.id]);
        first.import(&host).unwrap();
        assert_eq!(driver.pins.get(), 2);
    }

    #[test]
    fn a_refused_mapping_lets_go_of_its_pin() {
        let (driver, mut first, _) = devices();
        driver.refuse_mapping.set(true);
        let refused = first.import(&host());
        assert!(matches!(refused, Err(IoError::HostMemoryRefused { .. })));
        assert_eq!(driver.unpins.get(), 1);
    }

    #[test]
    fn a_refused_pin_is_refused_memory() {
        let (driver, mut first, _) = devices();
        driver.refuse_pin.set(true);
        let refused = first.import(&host());
        assert!(matches!(refused, Err(IoError::HostMemoryRefused { .. })));
    }

    #[test]
    fn a_failure_poisoning_the_device_is_no_refusal() {
        let failed = refused_unless_poisoned(DriverError::poisoned("map", 1), 1);
        assert!(!matches!(failed, IoError::HostMemoryRefused { .. }));
    }

    #[test]
    fn a_failed_wait_leaks_the_memory_pinned() {
        let (driver, mut first, _) = devices();
        let host = host();
        let handle = first.import(&host).unwrap();
        driver.done.set(Some(Err(DriverError::new("query", 1))));
        first.release_after_stream(&[handle.id]);
        assert_eq!(driver.unpins.get(), 0);
        assert_eq!(Arc::strong_count(host.memory()), 2);
    }

    #[test]
    fn a_wait_failed_by_a_poisoned_device_lets_go() {
        let (driver, mut first, _) = devices();
        let host = host();
        let handle = first.import(&host).unwrap();
        driver
            .done
            .set(Some(Err(DriverError::poisoned("query", 1))));
        first.release_after_stream(&[handle.id]);
        assert_eq!(driver.unpins.get(), 1);
        assert_eq!(Arc::strong_count(host.memory()), 1);
    }

    #[test]
    fn memory_that_fails_to_unpin_is_leaked() {
        let (driver, mut first, _) = devices();
        let host = host();
        let handle = first.import(&host).unwrap();
        driver.refuse_unpin.set(true);
        first.release_after_stream(&[handle.id]);
        assert_eq!(Arc::strong_count(host.memory()), 2);
    }

    #[test]
    fn memory_that_failed_to_unpin_imports_again() {
        let (driver, mut first, _) = devices();
        let host = host();
        let handle = first.import(&host).unwrap();
        driver.refuse_unpin.set(true);
        first.release_after_stream(&[handle.id]);
        driver.refuse_unpin.set(false);
        assert!(first.import(&host).is_ok());
    }

    #[test]
    fn dropping_a_device_lets_go_of_its_imports() {
        let (driver, mut first, _) = devices();
        let host = host();
        first.import(&host).unwrap();
        drop(first);
        assert_eq!(driver.unpins.get(), 1);
        assert_eq!(Arc::strong_count(host.memory()), 1);
    }
}
