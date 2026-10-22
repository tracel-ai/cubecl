//! A value a tile holds at its own element, read at the element its reader names.

use cubecl::ir::{ElemType, ExpandValue, Scope};
use cubecl::prelude::*;

use crate::Held;

/// `value`, held at `held`, as `E`: itself where the two are one element, cast where they differ.
pub(crate) fn read_as<E: Numeric>(
    scope: &Scope,
    value: ExpandValue,
    held: ElemType,
) -> NativeExpand<E> {
    if held == E::elem_type(scope) {
        return value.into();
    }
    let _held = scope.define_type::<Held>(held);
    E::__expand_cast_from(scope, NativeExpand::<Held>::from(value))
}
