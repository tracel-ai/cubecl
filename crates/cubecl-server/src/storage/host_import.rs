use super::{HostRange, StorageId};
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

/// The driver calls a runtime pins host memory for its devices with, and orders an import's
/// release behind its stream's work with.
pub trait HostPinner {
    /// Marks the work queued on the stream up to where it was recorded. Dropping it releases it.
    type Event;

    /// The memory this runtime pinned, which every device of the runtime shares.
    fn registry(&self) -> &'static HostRegistry;

    /// Pins `memory` and maps it for every device of the process. Fails for memory already
    /// pinned.
    fn pin(&self, memory: &Bytes) -> Result<(), DriverError>;

    /// Undoes [`pin`](Self::pin), once no device of the process reads `memory`.
    fn unpin(&self, memory: &Bytes) -> Result<(), DriverError>;

    /// This device's address of `host`, a byte of pinned memory.
    fn device_ptr(&self, host: *const u8) -> Result<u64, DriverError>;

    /// An event that completes once the work queued on the stream so far is done.
    fn record(&self) -> Result<Self::Event, DriverError>;

    /// Whether `event` completed.
    fn is_done(&self, event: &Self::Event) -> Result<bool, DriverError>;

    /// Waits for every stream of this device.
    fn synchronize(&self) -> Result<(), DriverError>;
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

    fn acquire(&self, pinner: &impl HostPinner, memory: &Bytes) -> Result<(), DriverError> {
        let key = Self::key(memory);
        let mut pins = self.pins.lock();
        if let Some((_, imports)) = pins.iter_mut().find(|(pinned, _)| *pinned == key) {
            *imports += 1;
            return Ok(());
        }
        pinner.pin(memory)?;
        pins.push((key, 1));
        Ok(())
    }

    fn release(&self, pinner: &impl HostPinner, memory: &Bytes) -> Result<(), DriverError> {
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
        pinner.unpin(memory)
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
pub struct HostImporter<P: HostPinner> {
    pinner: P,
    /// Keyed by storage id: the device's address of the import, and the memory it holds.
    imported: HashMap<StorageId, (u64, Arc<Bytes>)>,
    /// Released imports the stream may still read. Arrival: in stream order, so they complete in
    /// order and are let go of from the front.
    pending: VecDeque<PendingRelease<P::Event>>,
}

/// Imports released on a stream, let go of once the work queued there before is done. Another
/// stream's reads keep their bindings, so the memory is not released before those complete.
struct PendingRelease<E> {
    event: E,
    /// Arrival: let go of together once the event completes.
    released: Vec<Arc<Bytes>>,
}

impl<P: HostPinner> HostImporter<P> {
    /// No imports, pinning through `pinner`.
    pub fn new(pinner: P) -> Self {
        Self {
            pinner,
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
        self.pinner
            .registry()
            .acquire(&self.pinner, memory)
            .map_err(refused)?;
        let first = memory.as_ptr().wrapping_add(host.offset() as usize);
        let ptr = match self.pinner.device_ptr(first) {
            Ok(ptr) => ptr,
            Err(err) => {
                self.let_go(memory.clone());
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
            match self.pinner.record() {
                Ok(event) => self.pending.push_back(PendingRelease { event, released }),
                Err(err) => self.let_go_unread(released, &err),
            }
        }
        self.drop_completed();
    }

    /// Lets go of the imports whose release completed. Never while a stream records: the last
    /// release unpins the memory, and HIP fails a recording on the thread that pins or unpins.
    fn drop_completed(&mut self) {
        while let Some(release) = self.pending.front() {
            let done = self.pinner.is_done(&release.event);
            if matches!(done, Ok(false)) {
                return;
            }
            let Some(PendingRelease { released, .. }) = self.pending.pop_front() else {
                return;
            };
            match done {
                Ok(_) => released.into_iter().for_each(|memory| self.let_go(memory)),
                Err(err) => self.let_go_unread(released, &err),
            }
        }
    }

    /// Drops this import of `memory`, unpinning it when it was the last; leaked when unpinning
    /// fails, since a device may still read it.
    fn let_go(&self, memory: Arc<Bytes>) {
        if let Err(err) = self.pinner.registry().release(&self.pinner, &memory) {
            log::warn!("leaking host memory the driver may still read: {err}");
            core::mem::forget(memory);
        }
    }

    /// `released`, whose reads `failed` left unknown, let go of where the failure poisoned the
    /// device, whose work stopped with it, and leaked pinned otherwise: a kernel may still read it.
    fn let_go_unread(&self, released: Vec<Arc<Bytes>>, failed: &DriverError) {
        if failed.is_device_poisoned() {
            released.into_iter().for_each(|memory| self.let_go(memory));
            return;
        }
        log::warn!("leaking imported host memory a kernel may still read: {failed}");
        released.into_iter().for_each(core::mem::forget);
    }
}

impl<P: HostPinner> Drop for HostImporter<P> {
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
        match self.pinner.synchronize() {
            Ok(()) => released.into_iter().for_each(|memory| self.let_go(memory)),
            Err(err) => self.let_go_unread(released, &err),
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
    use super::*;
    use crate::storage::HOST_MEMORY_ALIGNMENT;
    use alloc::boxed::Box;
    use core::cell::Cell;

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

    struct Pinner {
        driver: &'static Driver,
        registry: &'static HostRegistry,
    }

    impl HostPinner for Pinner {
        type Event = ();

        fn registry(&self) -> &'static HostRegistry {
            self.registry
        }

        fn pin(&self, _memory: &Bytes) -> Result<(), DriverError> {
            self.driver.pins.set(self.driver.pins.get() + 1);
            Ok(())
        }

        fn unpin(&self, _memory: &Bytes) -> Result<(), DriverError> {
            if self.driver.refuse_unpin.get() {
                return Err(DriverError::new("unpin", 1));
            }
            self.driver.unpins.set(self.driver.unpins.get() + 1);
            Ok(())
        }

        fn device_ptr(&self, host: *const u8) -> Result<u64, DriverError> {
            match self.driver.refuse_mapping.get() {
                true => Err(DriverError::new("device_ptr", 1)),
                false => Ok(host as u64),
            }
        }

        fn record(&self) -> Result<(), DriverError> {
            Ok(())
        }

        fn is_done(&self, _event: &()) -> Result<bool, DriverError> {
            let done = self.driver.done.take().unwrap_or(Ok(true));
            self.driver.done.set(Some(done.clone()));
            done
        }

        fn synchronize(&self) -> Result<(), DriverError> {
            Ok(())
        }
    }

    /// Two devices' importers sharing one registry, as a runtime's devices do.
    fn devices() -> (&'static Driver, HostImporter<Pinner>, HostImporter<Pinner>) {
        let driver: &'static Driver = Box::leak(Box::default());
        let registry: &'static HostRegistry = Box::leak(Box::new(HostRegistry::new()));
        let importer = || HostImporter::new(Pinner { driver, registry });
        (driver, importer(), importer())
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
