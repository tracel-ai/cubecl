use cubecl_common::{e2m1, e2m1x2};
use cubecl_ir::{
    ConstantValue, ElemType, FloatKind, Scope,
    types::scalar::{Float4E2M1Type, Float4E2M1x2Type},
};
use pliron::r#type::TypeHandle;

use crate::prelude::*;

impl CubeType for e2m1 {
    type ExpandType = NativeExpand<e2m1>;
}

impl CubeDebug for e2m1 {}
impl Scalar for e2m1 {
    fn elem_type_native() -> ElemType {
        FloatKind::E2M1.into()
    }
}
impl CubePrimitive for e2m1 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        Float4E2M1Type::get(scope.ctx()).into()
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e2m1::from_f64(value)
    }
}

impl IntoRuntime for e2m1 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for e2m1 {
    type Expand = NativeExpand<e2m1>;
    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for e2m1 {}

impl CubeType for e2m1x2 {
    type ExpandType = NativeExpand<e2m1x2>;
}

impl CubeDebug for e2m1x2 {}
// Considered a scalar because it's really just a `u8` in a trenchcoat, and should be possible to
// store in a `Vector`.
impl Scalar for e2m1x2 {
    fn elem_type_native() -> ElemType {
        FloatKind::E2M1x2.into()
    }
}
impl CubePrimitive for e2m1x2 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        Float4E2M1x2Type::get(scope.ctx()).into()
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        let val = e2m1::from_f64(value).to_bits();
        // Fill both values, not sure this is ever useful but it works
        e2m1x2::from_bits(val | (val << 4))
    }
}

impl IntoRuntime for e2m1x2 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for e2m1x2 {
    type Expand = NativeExpand<e2m1x2>;
    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for e2m1x2 {}
