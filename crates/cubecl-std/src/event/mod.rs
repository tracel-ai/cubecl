use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};

#[derive(CubeType, Clone)]
/// This event bus allows users to trigger events at compilation time to modify the generated code.
///
/// # Warning
///
/// Recursion isn't supported with a runtime end condition, the compilation will fail with a
/// strange error.
pub struct ComptimeEventBus {
    #[allow(unused)]
    #[cube(comptime)]
    listener_family: Rc<RefCell<HashMap<TypeId, Vec<EventItem>>>>,
}

type EventItem = Box<dyn Any>;
type Call<E> =
    Box<dyn Fn(&mut Scope, &Box<dyn Any>, <E as CubeType>::ExpandType, ComptimeEventBusExpand)>;

struct Payload<E: CubeType> {
    listener: Box<dyn Any>,
    call: Call<E>,
}

impl Default for ComptimeEventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cube]
impl ComptimeEventBus {
    /// Creates a new event bus.
    pub fn new() -> Self {
        intrinsic!(|_| {
            ComptimeEventBusExpand {
                listener_family: Rc::new(RefCell::new(HashMap::new())),
            }
        })
    }

    #[allow(unused_variables)]
    /// Registers a new callback to be called when its event is launched.
    ///
    /// # Notes
    ///
    /// Multiple listeners for a single event type is supported. All the listeners will be called
    /// for each event in the same order they were registered.
    pub fn listener<L: EventListener>(&mut self, listener: L) {
        intrinsic!(|_| {
            let type_id = TypeId::of::<L::Event>();
            let mut listeners = self.listener_family.borrow_mut();

            // The call dynamic function erases the [EventListener] type.
            //
            // This is necessary since we need to clone the expand type when calling the expand
            // method. The listener is passed as a dynamic type and casted during the function call.
            let call =
                |scope: &mut cubecl::prelude::Scope,
                 listener: &Box<dyn Any>,
                 event: <L::Event as cubecl::prelude::CubeType>::ExpandType,
                 bus: <ComptimeEventBus as cubecl::prelude::CubeType>::ExpandType| {
                    let listener: &L::ExpandType = listener.downcast_ref().unwrap();
                    listener.clone().__expand_on_event_method(scope, event, bus)
                };
            let call: Call<L::Event> = Box::new(call);

            let listener: Box<dyn Any> = Box::new(listener);
            let payload = Payload::<L::Event> { listener, call };

            // Here we erase the event type, so that all listeners can be stored in the same event
            // bus which support multiple event types.
            let listener_dyn: Box<dyn Any> = Box::new(payload);

            match listeners.get_mut(&type_id) {
                Some(list) => list.push(listener_dyn),
                None => {
                    listeners.insert(type_id, vec![listener_dyn]);
                }
            }
        })
    }

    #[allow(unused_variables)]
    /// Registers a new event to be processed by [registered listeners](EventListener).
    pub fn event<E: CubeType + 'static>(&mut self, event: E) {
        intrinsic!(|scope| {
            let type_id = TypeId::of::<E>();
            let family = self.listener_family.borrow();
            let listeners = match family.get(&type_id) {
                Some(val) => val,
                None => return,
            };

            for listener in listeners.iter() {
                let payload = listener.downcast_ref::<Payload<E>>().unwrap();
                let call = &payload.call;
                call(scope, &payload.listener, event.clone(), self.clone());
            }
        })
    }
}

#[cube]
/// Defines a listener that is called each time an event is triggered on an
/// [event bus](ComptimeEventBus).
pub trait EventListener: 'static {
    /// The event type triggering the [EventListener::on_event] callback.
    type Event: CubeType + 'static;

    /// The function called when an event of the type [EventListener::Event] is registered on the
    /// [ComptimeEventBus].
    fn on_event(&mut self, event: Self::Event, bus: &mut ComptimeEventBus);
}
