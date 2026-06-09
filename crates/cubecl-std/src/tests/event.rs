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

#[derive(CubeType)]
pub struct EventListenerPosZero {
    items: Box<[f32]>,
}

#[derive(CubeType)]
pub struct EventListenerPosOne {
    items: Box<[f32]>,
}

#[derive(CubeType)]
pub struct EventListenerPosTwo {
    items: Box<[f32]>,
    times: ComptimeCell<Counter>,
}

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
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
        self.items[1] += f32::cast_from(event.value) * 2.0;
    }
}

#[cube]
impl EventListener for EventListenerPosTwo {
    type Event = EventFloat;

    fn on_event(&mut self, event: Self::Event, bus: &mut ComptimeEventBus) {
        comment!("On event pos two");
        self.items[2] += event.value;

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
fn test_1(items: &mut [f32]) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero {
        items: unsafe { items.as_boxed_unchecked() },
    };
    let listener_one = EventListenerPosOne {
        items: unsafe { items.as_boxed_unchecked() },
    };

    bus.listener::<EventListenerPosZero>(listener_zero);
    bus.listener::<EventListenerPosOne>(listener_one);

    bus.event::<EventUInt>(EventUInt { value: 5u32 });
}

#[cube]
fn test_2(items: &mut [f32]) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero {
        items: unsafe { items.as_boxed_unchecked() },
    };
    let listener_one = EventListenerPosOne {
        items: unsafe { items.as_boxed_unchecked() },
    };

    bus.listener::<EventListenerPosZero>(listener_zero);
    bus.listener::<EventListenerPosOne>(listener_one);

    bus.event::<EventUInt>(EventUInt { value: 15u32 });
}

#[cube]
fn test_3(items: &mut [f32]) {
    let mut bus = ComptimeEventBus::new();
    let listener_zero = EventListenerPosZero {
        items: unsafe { items.as_boxed_unchecked() },
    };
    let listener_one = EventListenerPosOne {
        items: unsafe { items.as_boxed_unchecked() },
    };
    let listener_two = EventListenerPosTwo {
        items: unsafe { items.as_boxed_unchecked() },
        times: ComptimeCell::new(Counter { value: 0u32 }),
    };

    bus.listener::<EventListenerPosZero>(listener_zero);
    bus.listener::<EventListenerPosOne>(listener_one);
    bus.listener::<EventListenerPosTwo>(listener_two);

    bus.event::<EventFloat>(EventFloat { value: 15.0f32 });
}

#[cube(launch_unchecked)]
fn launch_test_1(output: &mut [f32]) {
    output[0] = 0.0;
    output[1] = 0.0;
    test_1(output);
}

#[cube(launch_unchecked)]
fn launch_test_2(output: &mut [f32]) {
    output[0] = 0.0;
    output[1] = 0.0;
    test_2(output);
}

#[cube(launch_unchecked)]
fn launch_test_3(output: &mut [f32]) {
    output[0] = 0.0;
    output[1] = 0.0;
    output[2] = 0.0;
    test_3(output);
}

pub fn event_test_1<R: Runtime>(client: ComputeClient<R>) {
    let output = client.empty(8);

    unsafe {
        launch_test_1::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            BufferArg::from_raw_parts(output.clone(), 2),
        );
    }

    let bytes = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual, &[20.0, 50.0]);
}

pub fn event_test_2<R: Runtime>(client: ComputeClient<R>) {
    let output = client.empty(8);

    unsafe {
        launch_test_2::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            BufferArg::from_raw_parts(output.clone(), 2),
        )
    }

    let bytes = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual, &[15.0, 30.0]);
}

pub fn event_test_3<R: Runtime>(client: ComputeClient<R>) {
    let output = client.empty(12);

    unsafe {
        launch_test_3::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            BufferArg::from_raw_parts(output.clone(), 3),
        )
    }

    let bytes = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual, &[30.0, 900.0, 465.0]);
}

#[macro_export]
macro_rules! testgen_event {
    () => {
        mod event {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_1() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::event::event_test_1(client);
            }

            #[$crate::tests::test_log::test]
            fn test_2() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::event::event_test_2(client);
            }

            #[$crate::tests::test_log::test]
            fn test_3() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::event::event_test_3(client);
            }
        }
    };
}
