use std::vec::Vec;

use crate::device::{Device, DeviceId, DeviceService};

use super::*;

#[test]
fn test_concurrent_increment() {
    let device = TestDevice::<1>::new(0);
    let context = DeviceHandle::<TestDeviceState<1>>::new(device.to_id());

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
    let device_id = DeviceId {
        type_id: 0,
        index_id: 5,
    };
    let context = DeviceHandle::<TestDeviceState<1>>::new(device_id);
    let context_second = DeviceHandle::<TestDeviceState<2>>::new(device_id);

    context.submit(move |_state| {
        context_second.submit(move |_inner_state| {});
    });
}

#[derive(Debug, Clone, Default, new)]
/// Type is only to create different type ids.
pub struct TestDevice<const TYPE: u8> {
    index: u32,
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

    fn device_count(_type_id: u16) -> usize {
        TYPE as usize + 1
    }
}

impl<const T: usize> DeviceService for TestDeviceState<T> {
    fn init(_device_id: DeviceId) -> Self {
        TestDeviceState { counter: 0 }
    }
}
