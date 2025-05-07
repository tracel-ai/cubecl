use cubecl_common::e2m1;
use cubecl_ir::{Elem, ExpandElement, FloatKind, Scope};

use crate::prelude::{
    CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, IntoRuntime,
    into_mut_expand_element, into_runtime_expand_element,
};

impl CubeType for e2m1 {
    type ExpandType = ExpandElementTyped<e2m1>;
}

impl CubePrimitive for e2m1 {
    /// Return the element type to use on GPU
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Float(FloatKind::E2M1))
    }
}

impl IntoRuntime for e2m1 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementIntoMut for e2m1 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}
