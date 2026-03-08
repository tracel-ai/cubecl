use cubecl_common::{e4m3, e5m2, ue8m0};
use cubecl_ir::{ConstantValue, ElemType, FloatKind, Scope, Type};

use crate::prelude::*;

impl CubeType for e4m3 {
    type ExpandType = ExpandElementTyped<e4m3>;
}

impl CubePrimitive for e4m3 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::E4M3).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e4m3::from_f64(value)
    }
}

impl IntoRuntime for e4m3 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementAssign for e4m3 {}

impl CubeType for e5m2 {
    type ExpandType = ExpandElementTyped<e5m2>;
}

impl CubePrimitive for e5m2 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::E5M2).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e5m2::from_f64(value)
    }
}

impl IntoRuntime for e5m2 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementAssign for e5m2 {}

impl CubeType for ue8m0 {
    type ExpandType = ExpandElementTyped<ue8m0>;
}

impl CubePrimitive for ue8m0 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::UE8M0).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        ue8m0::from_f64(value)
    }
}

impl IntoRuntime for ue8m0 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementAssign for ue8m0 {}
