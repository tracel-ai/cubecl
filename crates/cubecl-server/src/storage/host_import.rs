use super::{HostRange, StorageId};
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
use cubecl_environment::sync::Mutex;

/// The driver calls a runtime pins host memory for its devices with. A release is ordered behind
/// its stream's work by the runtime's [`EventApi`].
pub trait HostMemoryApi: EventApi {
    /// The memory this runtime pinned, which every device of the runtime shares.
    fn host_registry() -> &'static HostRegistry;

    /// Pins `memory` and maps it for every device of the process. Fails for memory already
    /// pinned.
    ///
    /// # Safety
    ///
    /// `memory` stays allocated and in place until [`host_unregister`](Self::host_unregister).
    unsafe fn host_register(memory: &Bytes) -> Result<(), DriverError>;

    /// Undoes [`host_register`](Self::host_register).
    ///
    /// # Safety
    ///
    /// `memory` was registered, and no device of the process reads it any more.
    unsafe fn host_unregister(memory: &Bytes) -> Result<(), DriverError>;

    /// The current device's address of `host`, a byte of pinned memory.
    fn host_device_ptr(host: *const u8) -> Result<u64, DriverError>;

    /// Waits for every stream of the current device.
    fn device_synchronize() -> Result<(), DriverError>;
}

/// The host memory a runtime pinned, with how many imports of it its devices hold: the first
/// import pins it and the last release unpins it, since a driver pins a range once per process.
/// Memory that overlaps a pin without being it pins on its own, which the driver refuses.
pub struct HostRegistry {
    /// Keyed by the memory's address and length: the imports its devices hold. A handful of tables
    /// at most.
    pins: Mutex<Vec<((usize, usize), usize)>>,
}

impl HostRegistry {
    /// No memory pinned.
    pub const fn new() -> Self {
        Self {
            pins: Mutex::new(Vec::new()),
        }
    }

    fn acquire<A: HostMemoryApi>(&self, memory: &Bytes) -> Result<(), DriverError> {
        let key = Self::key(memory);
        let mut pins = self.pins.lock();
        if let Some((_, imports)) = pins.iter_mut().find(|(pinned, _)| *pinned == key) {
            *imports += 1;
            return Ok(());
        }
        // SAFETY: the import holds `memory` until its last release unregisters it.
        unsafe { A::host_register(memory)? };
        pins.push((key, 1));
        Ok(())
    }

    fn release<A: HostMemoryApi>(&self, memory: &Bytes) -> Result<(), DriverError> {
        let key = Self::key(memory);
        let mut pins = self.pins.lock();
        let at = pins
            .iter()
            .position(|(pinned, _)| *pinned == key)
            .expect("a released import should have been acquired");
        pins[at].1 -= 1;
        if pins[at].1 > 0 {
            return Ok(());
        }
        pins.swap_remove(at);
        // SAFETY: it pinned on its first import, and its last release runs once no device reads it.
        unsafe { A::host_unregister(memory) }
    }

    fn key(memory: &Bytes) -> (usize, usize) {
        (memory.as_ptr() as usize, memory.len())
    }
}

impl Default for HostRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Host memory a device reads in place, pinned while any device of the runtime imports it and
/// kept alive until the work queued before its release is done.
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
struct PendingRelease<A: EventApi> {
    event: Event<A>,
    /// Arrival: let go of together once the event completes.
    released: Vec<Arc<Bytes>>,
}

impl<A: HostMemoryApi> HostImporter<A> {
    /// No imports, released behind the work of `stream`, which outlives the importer.
    pub fn new(stream: A::Stream) -> Self {
        Self {
            stream,
            imported: HashMap::new(),
            pending: VecDeque::new(),
        }
    }

    /// Imports `host`, which the device reads until its id is released.
    ///
    /// # Errors
    ///
    /// [`IoError::HostMemoryRefused`] when the driver refuses to pin the memory or to map it for
    /// this device, and the driver's error when it poisons the device.
    pub fn import(&mut self, host: &HostRange) -> Result<StorageId, IoError> {
        // Imports released since the last flush unpin before this one pins more.
        self.drop_completed();
        let memory = host.memory();
        let refused = |err: DriverError| refused_unless_poisoned(err, memory.len());
        A::host_registry().acquire::<A>(memory).map_err(refused)?;
        let first = memory.as_ptr().wrapping_add(host.offset() as usize);
        let ptr = match A::host_device_ptr(first) {
            Ok(ptr) => ptr,
            Err(err) => {
                Self::let_go(memory.clone());
                return Err(refused(err));
            }
        };
        let id = StorageId::new();
        self.imported.insert(id, (ptr, memory.clone()));
        Ok(id)
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
        if !released.is_empty() {
            match self.record() {
                Ok(event) => self.pending.push_back(PendingRelease { event, released }),
                Err(err) => Self::let_go_unread(released, &err),
            }
        }
        self.drop_completed();
    }
}

impl<A: HostMemoryApi> HostImporter<A> {
    fn record(&self) -> Result<Event<A>, DriverError> {
        let event = Event::new()?;
        event.record(self.stream)?;
        Ok(event)
    }

    /// Lets go of the imports whose release completed. Never while a stream records: the last
    /// release unpins the memory, and HIP fails a recording on the thread that pins or unpins.
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
        if let Err(err) = A::host_registry().release::<A>(&memory) {
            log::warn!("leaking host memory the driver may still read: {err}");
            core::mem::forget(memory);
        }
    }

    /// `released`, whose reads `failed` left unknown, let go of where the failure poisoned the
    /// device, whose work stopped with it, and leaked pinned otherwise: a kernel may still read it.
    fn let_go_unread(released: Vec<Arc<Bytes>>, failed: &DriverError) {
        if failed.is_device_poisoned() {
            released.into_iter().for_each(Self::let_go);
            return;
        }
        log::warn!("leaking imported host memory a kernel may still read: {failed}");
        released.into_iter().for_each(core::mem::forget);
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
    extern crate std;

    use super::*;
    use crate::storage::HOST_MEMORY_ALIGNMENT;
    use alloc::boxed::Box;
    use cubecl_common::profile::Duration;
    use std::cell::Cell;

    /// Counts the driver calls, and answers as the test sets it.
    #[derive(Default)]
    struct Driver {
        pins: Cell<usize>,
        unpins: Cell<usize>,
        refuse_mapping: Cell<bool>,
        refuse_unpin: Cell<bool>,
        /// Whether a recorded event completed, or the error its query returns.
        done: Cell<Option<Result<bool, DriverError>>>,
    }

    std::thread_local! {
        /// Replaced by each test's [`devices`], since tests may share a thread.
        static DRIVER: Cell<&'static Driver> = Cell::new(Box::leak(Box::default()));
        static REGISTRY: Cell<&'static HostRegistry> =
            Cell::new(Box::leak(Box::new(HostRegistry::new())));
    }

    fn driver() -> &'static Driver {
        DRIVER.with(Cell::get)
    }

    /// A runtime whose devices are this thread's [`Driver`].
    struct Fake;

    impl EventApi for Fake {
        type Event = ();
        type Stream = ();

        const BACKEND: &'static str = "fake";

        fn event_create() -> Result<(), DriverError> {
            Ok(())
        }

        fn event_destroy(_event: &mut ()) -> Result<(), DriverError> {
            Ok(())
        }

        fn event_record(_event: &(), _stream: ()) -> Result<(), DriverError> {
            Ok(())
        }

        fn event_query(_event: &()) -> Result<bool, DriverError> {
            let done = driver().done.take().unwrap_or(Ok(true));
            driver().done.set(Some(done.clone()));
            done
        }

        fn event_wait(_event: &()) -> Result<(), DriverError> {
            unreachable!("the importer should never block on an event")
        }

        fn event_elapsed(_start: &(), _end: &()) -> Result<Duration, DriverError> {
            unreachable!("the importer should never time events")
        }

        fn stream_wait_event(_stream: (), _event: &()) -> Result<(), DriverError> {
            unreachable!("the importer should never order streams")
        }

        fn stream_synchronize(_stream: ()) -> Result<(), DriverError> {
            unreachable!("the importer should synchronize the device")
        }

        fn stream_create_non_blocking() -> Result<(), DriverError> {
            unreachable!("the importer should never create a stream")
        }

        fn stream_destroy(_stream: ()) -> Result<(), DriverError> {
            unreachable!("the importer should never destroy a stream")
        }
    }

    impl HostMemoryApi for Fake {
        fn host_registry() -> &'static HostRegistry {
            REGISTRY.with(Cell::get)
        }

        unsafe fn host_register(_memory: &Bytes) -> Result<(), DriverError> {
            driver().pins.set(driver().pins.get() + 1);
            Ok(())
        }

        unsafe fn host_unregister(_memory: &Bytes) -> Result<(), DriverError> {
            if driver().refuse_unpin.get() {
                return Err(DriverError::new("unpin", 1));
            }
            driver().unpins.set(driver().unpins.get() + 1);
            Ok(())
        }

        fn host_device_ptr(host: *const u8) -> Result<u64, DriverError> {
            match driver().refuse_mapping.get() {
                true => Err(DriverError::new("device_ptr", 1)),
                false => Ok(host as u64),
            }
        }

        fn device_synchronize() -> Result<(), DriverError> {
            Ok(())
        }
    }

    /// Two devices' importers sharing one registry, as a runtime's devices do.
    fn devices() -> (&'static Driver, HostImporter<Fake>, HostImporter<Fake>) {
        DRIVER.with(|driver| driver.set(Box::leak(Box::default())));
        REGISTRY.with(|registry| registry.set(Box::leak(Box::new(HostRegistry::new()))));
        (driver(), HostImporter::new(()), HostImporter::new(()))
    }

    fn host() -> HostRange {
        let bytes = Bytes::zeroed_aligned(HOST_MEMORY_ALIGNMENT, HOST_MEMORY_ALIGNMENT).unwrap();
        HostRange::new(Arc::new(bytes), 0..HOST_MEMORY_ALIGNMENT as u64).unwrap()
    }

    #[test]
    fn devices_importing_one_memory_pin_it_once() {
        let (driver, mut first, mut second) = devices();
        let host = host();
        first.import(&host).unwrap();
        second.import(&host).unwrap();
        assert_eq!(driver.pins.get(), 1);
    }

    #[test]
    fn memory_overlapping_a_pin_pins_on_its_own() {
        let (driver, mut first, _) = devices();
        let bytes =
            Bytes::zeroed_aligned(2 * HOST_MEMORY_ALIGNMENT, HOST_MEMORY_ALIGNMENT).unwrap();
        let whole = Arc::new(bytes.shared());
        let prefix = Arc::new(whole.view(0, HOST_MEMORY_ALIGNMENT).unwrap());
        let chunk = HOST_MEMORY_ALIGNMENT as u64;
        first
            .import(&HostRange::new(prefix, 0..chunk).unwrap())
            .unwrap();
        first
            .import(&HostRange::new(whole, 0..2 * chunk).unwrap())
            .unwrap();
        assert_eq!(driver.pins.get(), 2);
    }

    #[test]
    fn the_last_release_unpins() {
        let (driver, mut first, mut second) = devices();
        let host = host();
        let one = first.import(&host).unwrap();
        let other = second.import(&host).unwrap();
        first.release_after_stream(&[one]);
        assert_eq!(driver.unpins.get(), 0);
        second.release_after_stream(&[other]);
        assert_eq!(driver.unpins.get(), 1);
    }

    #[test]
    fn a_release_waits_for_the_stream() {
        let (driver, mut first, _) = devices();
        let host = host();
        let id = first.import(&host).unwrap();
        driver.done.set(Some(Ok(false)));
        first.release_after_stream(&[id]);
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
        let id = first.import(&host).unwrap();
        first.release_after_stream(&[id]);
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
    fn a_failed_wait_leaks_the_memory_pinned() {
        let (driver, mut first, _) = devices();
        let host = host();
        let id = first.import(&host).unwrap();
        driver.done.set(Some(Err(DriverError::new("query", 1))));
        first.release_after_stream(&[id]);
        assert_eq!(driver.unpins.get(), 0);
        assert_eq!(Arc::strong_count(host.memory()), 2);
    }

    #[test]
    fn a_wait_failed_by_a_poisoned_device_lets_go() {
        let (driver, mut first, _) = devices();
        let host = host();
        let id = first.import(&host).unwrap();
        driver
            .done
            .set(Some(Err(DriverError::poisoned("query", 1))));
        first.release_after_stream(&[id]);
        assert_eq!(driver.unpins.get(), 1);
        assert_eq!(Arc::strong_count(host.memory()), 1);
    }

    #[test]
    fn memory_that_fails_to_unpin_is_leaked() {
        let (driver, mut first, _) = devices();
        let host = host();
        let id = first.import(&host).unwrap();
        driver.refuse_unpin.set(true);
        first.release_after_stream(&[id]);
        assert_eq!(Arc::strong_count(host.memory()), 2);
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
