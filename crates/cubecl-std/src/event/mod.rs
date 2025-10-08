use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};

#[derive(CubeType, Clone)]
pub struct EventBus {
    #[cube(comptime)]
    listeners: Rc<RefCell<HashMap<TypeId, EventItem>>>,
}

type EventItem = Rc<RefCell<Box<dyn Any>>>;
type Call<E> = Box<dyn Fn(&mut Scope, &Box<dyn Any>, <E as CubeType>::ExpandType, EventBusExpand)>;

struct Payload<E: CubeType> {
    listener: Box<dyn Any>,
    call: Call<E>,
}

#[cube]
#[allow(unused_variables)]
impl EventBus {
    pub fn new() -> Self {
        intrinsic!(|_| {
            EventBusExpand {
                listeners: Rc::new(RefCell::new(HashMap::new())),
            }
        })
    }

    pub fn listener<L: EventListener>(&mut self, listener: L) {
        intrinsic!(|_| {
            let type_id = TypeId::of::<L::Event>();
            let mut listeners = self.listeners.borrow_mut();
            let call =
                |scope: &mut cubecl::prelude::Scope,
                 data: &Box<dyn Any>,
                 event: <L::Event as cubecl::prelude::CubeType>::ExpandType,
                 bus: <EventBus as cubecl::prelude::CubeType>::ExpandType| {
                    let listener: &L::ExpandType = data.downcast_ref().unwrap();
                    listener.clone().__expand_on_event_method(scope, event, bus)
                };

            let call: Call<L::Event> = Box::new(call);
            let data: Box<dyn Any> = Box::new(listener);

            let payload = Payload::<L::Event> {
                listener: data,
                call,
            };

            println!("{}", core::any::type_name_of_val(&payload));
            let listener_dyn: Box<dyn Any> = Box::new(payload);
            let listener_rc = Rc::new(RefCell::new(listener_dyn));
            listeners.insert(type_id, listener_rc);
        })
    }

    pub fn event<E: CubeType + 'static>(&mut self, event: E) {
        intrinsic!(|scope| {
            let type_id = TypeId::of::<E>();
            let mut listeners = self.listeners.borrow();
            let listener = listeners
                .get(&type_id)
                .expect("Should have a listener registered for the event")
                .clone();
            core::mem::drop(listeners);

            let listener_ref = listener.borrow();
            println!("{}", core::any::type_name::<Payload<E>>());
            let payload = listener_ref.downcast_ref::<Payload<E>>().unwrap();
            let call = &payload.call;
            call(scope, &payload.listener, event, self);
        })
    }
}

#[cube]
pub trait EventListener: 'static {
    type Event: CubeType + 'static;
    fn on_event(&mut self, event: Self::Event, bus: &mut EventBus);
}
