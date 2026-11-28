//! A runtime whose driver the host memory tests set up and count.

extern crate std;

use super::{HostImports, HostMemoryApi, HostPins, HostRange};
use crate::device_events::EventApi;
use crate::driver::DriverError;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::profile::Duration;
use cubecl_environment::bytes::Bytes;
use std::cell::{Cell, RefCell};

use super::HOST_MEMORY_ALIGNMENT_BYTES;

/// Counts the driver calls, and answers as the test sets it.
#[derive(Default)]
pub struct Driver {
    pub pins: Cell<usize>,
    pub unpins: Cell<usize>,
    /// Arrival: the memory registered, by address and length, which a second registration of
    /// refuses as a driver does.
    pub held: RefCell<Vec<(usize, usize)>>,
    pub refuse_pin: Cell<bool>,
    pub refuse_mapping: Cell<bool>,
    pub refuse_unpin: Cell<bool>,
    /// Whether a recorded event completed, or the error its query returns.
    pub done: Cell<Option<Result<bool, DriverError>>>,
}

std::thread_local! {
    /// Replaced by each test's [`devices`], since tests may share a thread.
    static DRIVER: Cell<&'static Driver> = Cell::new(Box::leak(Box::default()));
    static PINS: Cell<&'static HostPins> = Cell::new(Box::leak(Box::new(HostPins::new())));
}

pub fn driver() -> &'static Driver {
    DRIVER.with(Cell::get)
}

/// A runtime whose devices are this thread's [`Driver`].
pub struct Fake;

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

fn key(memory: &Bytes) -> (usize, usize) {
    (memory.as_ptr() as usize, memory.len())
}

impl HostMemoryApi for Fake {
    fn host_pins() -> &'static HostPins {
        PINS.with(Cell::get)
    }

    unsafe fn host_register(memory: &Bytes) -> Result<(), DriverError> {
        let driver = driver();
        if driver.refuse_pin.get() || driver.held.borrow().contains(&key(memory)) {
            return Err(DriverError::new("register", 1));
        }
        driver.held.borrow_mut().push(key(memory));
        driver.pins.set(driver.pins.get() + 1);
        Ok(())
    }

    unsafe fn host_unregister(memory: &Bytes) -> Result<(), DriverError> {
        let driver = driver();
        if driver.refuse_unpin.get() {
            return Err(DriverError::new("unpin", 1));
        }
        driver.held.borrow_mut().retain(|held| *held != key(memory));
        driver.unpins.set(driver.unpins.get() + 1);
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

/// A fresh driver, and two devices' imports sharing one runtime's pins, as a runtime's devices do.
pub fn devices() -> (&'static Driver, HostImports<Fake>, HostImports<Fake>) {
    DRIVER.with(|driver| driver.set(Box::leak(Box::default())));
    PINS.with(|pins| pins.set(Box::leak(Box::new(HostPins::new()))));
    // SAFETY: the fake stream is `()`, which nothing destroys.
    let imports = || unsafe { HostImports::on_stream(()) };
    (driver(), imports(), imports())
}

/// One chunk of aligned host memory, whole.
pub fn host() -> HostRange {
    let bytes =
        Bytes::zeroed_aligned(HOST_MEMORY_ALIGNMENT_BYTES, HOST_MEMORY_ALIGNMENT_BYTES).unwrap();
    HostRange::new(Arc::new(bytes), 0..HOST_MEMORY_ALIGNMENT_BYTES as u64).unwrap()
}
