use crate as cubecl;
use cubecl::prelude::*;
use cubecl_macros::intrinsic;

#[cube]
#[allow(unused_variables)]
/// Push a validation error that will make the kernel compilation to fail.
///
/// # Notes
///
/// The error can be catched after the kernel is launched.
pub fn push_validation_error(#[comptime] msg: String) {
    intrinsic! {|scope| scope.push_error(msg)}
}
