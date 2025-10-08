use crate::event::{ComptimeEventBus, EventListener, EventListenerExpand};
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
pub struct DummyEvent {
    #[cube(comptime)]
    pub value: u32,
}

#[derive(CubeType, Clone)]
pub struct EventListenerPosZero {
    items: SliceMut<f32>,
}

#[derive(CubeType, Clone)]
pub struct EventListenerPosOne {
    items: SliceMut<f32>,
}

#[cube]
impl EventListener for EventListenerPosZero {
    type Event = DummyEvent;

    fn on_event(&mut self, event: Self::Event, bus: &mut ComptimeEventBus) {
        if comptime!(event.value < 10) {
            comment!("On event pos zero < 10");
            bus.event::<DummyEvent>(DummyEvent {
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
    type Event = DummyEvent;

    fn on_event(&mut self, event: Self::Event, _bus: &mut ComptimeEventBus) {
        comment!("On event pos one");
        self.items[1] = (f32::cast_from(event.value) * 2.0) + self.items[1];
    }
}

#[cube]
fn test_1(items: SliceMut<f32>) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero { items };
    bus.listener::<EventListenerPosZero>(listener_zero);

    let listener_one = EventListenerPosOne { items };
    bus.listener::<EventListenerPosOne>(listener_one);

    bus.event::<DummyEvent>(DummyEvent { value: 5u32 });
}

#[cube]
fn test_2(items: SliceMut<f32>) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero { items };
    bus.listener::<EventListenerPosZero>(listener_zero);

    let listener_one = EventListenerPosOne { items };
    bus.listener::<EventListenerPosOne>(listener_one);

    bus.event::<DummyEvent>(DummyEvent { value: 15u32 });
}

#[cube(launch_unchecked)]
fn launch_test_1(output: &mut Array<f32>) {
    test_1(output.to_slice_mut());
}

#[cube(launch_unchecked)]
fn launch_test_2(output: &mut Array<f32>) {
    test_2(output.to_slice_mut());
}

pub fn event_test_1<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
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

pub fn event_test_2<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
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
        }
    };
}
