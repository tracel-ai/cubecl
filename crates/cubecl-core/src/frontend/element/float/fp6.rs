use cubecl_common::{e2m3, e3m2};
use cubecl_ir::{ConstantValue, ElemType, ExpandElement, FloatKind, Scope, StorageType};

use crate::prelude::{
    CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, IntoRuntime,
    into_mut_expand_element, into_runtime_expand_element,
};

impl CubeType for e2m3 {
    type ExpandType = ExpandElementTyped<e2m3>;
}

impl CubePrimitive for e2m3 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<StorageType> {
        Some(ElemType::Float(FloatKind::E2M3).into())
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("e2m3 doesn't yet support conversion");
    }
}

impl IntoRuntime for e2m3 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementIntoMut for e2m3 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl CubeType for e3m2 {
    type ExpandType = ExpandElementTyped<e3m2>;
}

impl CubePrimitive for e3m2 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<StorageType> {
        Some(ElemType::Float(FloatKind::E3M2).into())
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("e3m2 doesn't yet support conversion");
    }
}

impl IntoRuntime for e3m2 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementIntoMut for e3m2 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}
