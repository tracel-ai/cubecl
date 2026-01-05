use crate as cubecl;
use alloc::rc::Rc;
use cubecl::prelude::*;
use cubecl_ir::DeviceProperties;
use cubecl_macros::intrinsic;

///
#[cube]
#[allow(unused_variables)]
pub fn device_properties() -> comptime_type!(Rc<DeviceProperties>) {
    intrinsic!(|scope| scope.properties.as_ref().unwrap().clone())
}
