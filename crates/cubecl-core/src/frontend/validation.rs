use crate as cubecl;
use alloc::{format, string::String};
use cubecl::prelude::*;
use cubecl_ir::{ElemType, Scope, features::ComplexUsage};
use cubecl_macros::intrinsic;

#[cube]
/// Push a validation error that will make kernel compilation fail.
pub fn push_validation_error(#[comptime] msg: String) {
    intrinsic! {|scope| scope.push_error(msg)}
}

pub(crate) fn require_complex_usage(
    scope: &Scope,
    ty: ElemType,
    usage: ComplexUsage,
    op_name: &'static str,
) {
    if !ty.is_complex() {
        return;
    }
    let Some(properties) = scope.state().device_properties.clone() else {
        return;
    };
    if !properties.supports_complex_usage(ty, usage) {
        scope.push_error(format!(
            "Complex operation `{op_name}` requires {usage:?} support for `{ty}`, but the active runtime does not advertise it."
        ));
    }
}
