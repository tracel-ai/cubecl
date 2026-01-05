use cubecl_common::{e4m3, e5m2, ue8m0};
use cubecl_ir::{ConstantValue, ElemType, ExpandElement, FloatKind, Scope, StorageType};

use crate::prelude::{
    CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, IntoRuntime, Numeric,
    into_mut_expand_element, into_runtime_expand_element,
};

impl CubeType for e4m3 {
    type ExpandType = ExpandElementTyped<e4m3>;
}

impl CubePrimitive for e4m3 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<StorageType> {
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

impl Numeric for e4m3 {
    fn min_value() -> Self {
        Self::from_f64(Self::MIN)
    }
    fn max_value() -> Self {
        Self::from_f64(Self::MAX)
    }
}

impl ExpandElementIntoMut for e4m3 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl CubeType for e5m2 {
    type ExpandType = ExpandElementTyped<e5m2>;
}

impl CubePrimitive for e5m2 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<StorageType> {
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

impl Numeric for e5m2 {
    fn min_value() -> Self {
        Self::from_f64(Self::MIN)
    }
    fn max_value() -> Self {
        Self::from_f64(Self::MAX)
    }
}

impl ExpandElementIntoMut for e5m2 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl CubeType for ue8m0 {
    type ExpandType = ExpandElementTyped<ue8m0>;
}

impl CubePrimitive for ue8m0 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<StorageType> {
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

impl Numeric for ue8m0 {
    fn min_value() -> Self {
        Self::from_f64(Self::MIN)
    }
    fn max_value() -> Self {
        Self::from_f64(Self::MAX)
    }
}

impl ExpandElementIntoMut for ue8m0 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}
