use crate as cubecl;
use alloc::sync::Arc;
use cubecl::prelude::*;
use cubecl_ir::{DeviceProperties, HardwareProperties, TargetProperties};
use cubecl_macros::intrinsic;

/// Retrieves the [`device_properties`](DeviceProperties).
#[cube]
pub fn device_properties() -> comptime_type!(Arc<DeviceProperties>) {
    intrinsic!(|scope| scope.state().device_properties.as_ref().unwrap().clone())
}

/// Retrieves the [`target_properties`](TargetProperties): how the compiler lays out what it
/// emits for the device, such as a matrix instruction's registers.
#[cube]
pub fn target_properties() -> comptime_type!(TargetProperties) {
    intrinsic!(|scope| scope.state().target_properties.clone())
}

/// Retrieves the [`hardware_properties`](HardwareProperties).
#[cube]
pub fn hardware_properties() -> comptime_type!(HardwareProperties) {
    let props = &device_properties().comptime().hardware;
    props.clone()
}
