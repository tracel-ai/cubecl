use std::vec::Vec;

use super::*;
use crate::device::{Device, DeviceId, DeviceService};
use cubecl_environment::sync::Arc;

#[test]
fn test_concurrent_increment() {
    let context = DeviceFixture::new(
        DeviceHandle::<TestDeviceState<1>>::new,
        DeviceHandle::<TestDeviceState<1>>::shutdown,
    );

    let thread_count = 10;
    let mut handles = Vec::new();

    for _ in 0..thread_count {
        let ctx = context.clone();
        handles.push(std::thread::spawn(move || {
            ctx.submit(|state| {
                state.counter += 1;
            });
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let count = context.submit_blocking(move |state| state.counter).unwrap();
    assert_eq!(count, thread_count);
}
#[test]
fn test_recursive_execution_different_state() {
    let context = DeviceFixture::new(
        DeviceHandle::<TestDeviceState<1>>::new,
        DeviceHandle::<TestDeviceState<1>>::shutdown,
    );
    // A second service on the same device. Declared after the fixture, so it drops
    // before the shutdown the fixture runs.
    let context_second = DeviceHandle::<TestDeviceState<2>>::new(context.device_id());

    context.submit(move |_state| {
        context_second.submit(move |_inner_state| {});
    });
}

#[derive(Debug, Clone, Default, new)]
/// Type is only to create different type ids. Device ids come from the test fixture,
/// so this only exists to keep the [`Device`] implementation below compiling.
#[allow(dead_code)]
pub struct TestDevice<const TYPE: u8> {
    index: u16,
}

pub struct TestDeviceState<const T: usize> {
    counter: usize,
}

impl<const TYPE: u8> Device for TestDevice<TYPE> {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index,
        }
    }
}

impl<const T: usize> DeviceService for TestDeviceState<T> {
    fn init(_device_id: DeviceId) -> Self {
        TestDeviceState { counter: 0 }
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }
}
