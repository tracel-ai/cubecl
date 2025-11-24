use crate::event::{ComptimeEventBus, EventListener, EventListenerExpand};
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
pub struct EventUInt {
    #[cube(comptime)]
    pub value: u32,
}

#[derive(CubeType)]
pub struct EventFloat {
    #[cube(comptime)]
    pub value: f32,
}

#[derive(CubeType, Clone)]
pub struct EventListenerPosZero {
    items: SliceMut<f32>,
}

#[derive(CubeType, Clone)]
pub struct EventListenerPosOne {
    items: SliceMut<f32>,
}

#[derive(CubeType, Clone)]
pub struct EventListenerPosTwo {
    items: SliceMut<f32>,
    times: ComptimeCell<Counter>,
}

#[derive(CubeType, Clone)]
pub struct Counter {
    #[cube(comptime)]
    value: u32,
}

#[cube]
impl EventListener for EventListenerPosZero {
    type Event = EventUInt;

    fn on_event(&mut self, event: Self::Event, bus: &mut ComptimeEventBus) {
        if comptime!(event.value < 10) {
            comment!("On event pos zero < 10");
            bus.event::<EventUInt>(EventUInt {
                value: comptime!(15u32 + event.value),
            });
        } else {
            comment!("On event pos zero >= 10");
            self.items[0] = f32::cast_from(event.value);
        }
    }
}

#[cube]
impl EventListener for EventListenerPosOne {
    type Event = EventUInt;

    fn on_event(&mut self, event: Self::Event, _bus: &mut ComptimeEventBus) {
        comment!("On event pos one");
        self.items[1] = (f32::cast_from(event.value) * 2.0) + self.items[1];
    }
}

#[cube]
impl EventListener for EventListenerPosTwo {
    type Event = EventFloat;

    fn on_event(&mut self, event: Self::Event, bus: &mut ComptimeEventBus) {
        comment!("On event pos two");
        self.items[2] = event.value + self.items[2];

        let times = self.times.read();
        self.times.store(Counter {
            value: comptime!(times.value + 1),
        });

        if comptime!(times.value < 4) {
            bus.event::<EventFloat>(EventFloat {
                value: comptime!(event.value * 2.0),
            });
            bus.event::<EventUInt>(EventUInt {
                value: comptime!((event.value * 2.0) as u32),
            });
        }
    }
}

#[cube]
fn test_1(items: SliceMut<f32>) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero { items };
    let listener_one = EventListenerPosOne { items };

    bus.listener::<EventListenerPosZero>(listener_zero);
    bus.listener::<EventListenerPosOne>(listener_one);

    bus.event::<EventUInt>(EventUInt { value: 5u32 });
}

#[cube]
fn test_2(items: SliceMut<f32>) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero { items };
    let listener_one = EventListenerPosOne { items };

    bus.listener::<EventListenerPosZero>(listener_zero);
    bus.listener::<EventListenerPosOne>(listener_one);

    bus.event::<EventUInt>(EventUInt { value: 15u32 });
}

#[cube]
fn test_3(items: SliceMut<f32>) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero { items };
    let listener_one = EventListenerPosOne { items };
    let listener_two = EventListenerPosTwo {
        items,
        times: ComptimeCell::new(Counter { value: 0u32 }),
    };

    bus.listener::<EventListenerPosZero>(listener_zero);
    bus.listener::<EventListenerPosOne>(listener_one);
    bus.listener::<EventListenerPosTwo>(listener_two);

    bus.event::<EventFloat>(EventFloat { value: 15.0f32 });
}

#[cube(launch_unchecked)]
fn launch_test_1(output: &mut Array<f32>) {
    output[0] = 0.0;
    output[1] = 0.0;
    test_1(output.to_slice_mut());
}

#[cube(launch_unchecked)]
fn launch_test_2(output: &mut Array<f32>) {
    output[0] = 0.0;
    output[1] = 0.0;
    test_2(output.to_slice_mut());
}

#[cube(launch_unchecked)]
fn launch_test_3(output: &mut Array<f32>) {
    output[0] = 0.0;
    output[1] = 0.0;
    output[2] = 0.0;
    test_3(output.to_slice_mut());
}

pub fn event_test_1<R: Runtime>(client: ComputeClient<R::Server>) {
    let output = client.empty(8);

    unsafe {
        launch_test_1::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            ArrayArg::from_raw_parts::<f32>(&output, 2, 1),
        );
    }

    let bytes = client.read_one(output);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual, &[20.0, 50.0]);
}

pub fn event_test_2<R: Runtime>(client: ComputeClient<R::Server>) {
    let output = client.empty(8);

    unsafe {
        launch_test_2::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            ArrayArg::from_raw_parts::<f32>(&output, 2, 1),
        );
    }

    let bytes = client.read_one(output);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual, &[15.0, 30.0]);
}

pub fn event_test_3<R: Runtime>(client: ComputeClient<R::Server>) {
    let output = client.empty(12);

    unsafe {
        launch_test_3::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            ArrayArg::from_raw_parts::<f32>(&output, 3, 1),
        );
    }

    let bytes = client.read_one(output);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual, &[30.0, 900.0, 465.0]);
}

#[macro_export]
macro_rules! testgen_event {
    () => {
        mod event {
            use super::*;

            #[test]
            fn test_1() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::event::event_test_1::<TestRuntime>(client);
            }

            #[test]
            fn test_2() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::event::event_test_2::<TestRuntime>(client);
            }

            #[test]
            fn test_3() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::event::event_test_3::<TestRuntime>(client);
            }
        }
    };
}
