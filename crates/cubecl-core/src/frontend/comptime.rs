use crate as cubecl;
use alloc::rc::Rc;
use cubecl::prelude::*;
use cubecl_ir::{DeviceProperties, HardwareProperties};
use cubecl_macros::intrinsic;

/// Retrieves the [`device_properties`](DeviceProperties).
#[cube]
#[allow(unused_variables)]
pub fn device_properties() -> comptime_type!(Rc<DeviceProperties>) {
    intrinsic!(|scope| scope.properties.as_ref().unwrap().clone())
}

/// Retrieves the [`hardware_properties`](HardwareProperties).
#[cube]
#[allow(unused_variables)]
pub fn hardware_properties() -> comptime_type!(HardwareProperties) {
    let props = &device_properties().comptime().hardware;
    comptime!(props.clone())
}
