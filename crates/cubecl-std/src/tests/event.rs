use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::event::{EventBus, EventListener, EventListenerExpand};

use super::*;

#[derive(CubeType)]
pub struct Event1 {
    #[cube(comptime)]
    pub index: u32,
}

#[derive(CubeType, Clone)]
pub struct EventListener1 {
    items: SliceMut<f32>,
}

#[cube]
impl EventListener for EventListener1 {
    type Event = Event1;

    fn on_event(&mut self, event: Self::Event, bus: &mut EventBus) {
        self.items[0] = f32::cast_from(event.index);

        if comptime!(event.index < 10) {
            bus.event::<Event1>(Event1 { index: 15u32 });
        }
    }
}

#[cube]
fn allo(items: SliceMut<f32>) {
    let mut bus = EventBus::new();
    let listener = EventListener1 { items };
    bus.listener::<EventListener1>(listener);
    bus.event::<Event1>(Event1 { index: 5u32 });
}

#[cube(launch_unchecked)]
pub fn launch_allo(output: &mut Array<f32>) {
    allo(output.to_slice_mut());
}

pub fn test<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let output = client.empty(4);

    unsafe {
        launch_allo::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            ArrayArg::from_raw_parts::<f32>(&output, 32, 1),
        );
    }

    let bytes = client.read_one(output);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual, &[6.0])
}

#[macro_export]
macro_rules! testgen_event {
    () => {
        mod event {
            use super::*;

            #[test]
            fn test_event() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::event::test::<TestRuntime>(client);
            }
        }
    };
}
