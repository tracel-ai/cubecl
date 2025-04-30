use cubecl_ir::{ExpandElement, Scope};

use crate::frontend::{CubePrimitive, CubeType};
use crate::ir::Elem;

use super::{
    ExpandElementIntoMut, ExpandElementTyped, IntoMut, IntoRuntime, into_mut_expand_element,
};

/// Extension trait for [bool].
pub trait BoolOps {
    #[allow(clippy::new_ret_no_self)]
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(
        _scope: &mut Scope,
        value: ExpandElementTyped<bool>,
    ) -> ExpandElementTyped<bool> {
        ExpandElement::Plain(Elem::Bool.from_constant(*value.expand)).into()
    }
}

impl BoolOps for bool {}

impl CubeType for bool {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for bool {
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Bool)
    }
}

impl IntoRuntime for bool {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        IntoMut::into_mut(expand, scope, false)
    }
}

impl ExpandElementIntoMut for bool {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement, is_mut: bool) -> ExpandElement {
        into_mut_expand_element(scope, elem, is_mut)
    }
}
