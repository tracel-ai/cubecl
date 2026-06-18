use cubecl_common::{e4m3, e5m2, ue8m0};
use cubecl_ir::{
    ConstantValue, FloatKind, Scope,
    types::scalar::{Float8E4M3Type, Float8E5M2Type, Float8E8M0Type},
};
use pliron::r#type::TypeHandle;

use crate::prelude::*;

impl CubeType for e4m3 {
    type ExpandType = NativeExpand<e4m3>;
}

impl CubeDebug for e4m3 {}
impl Scalar for e4m3 {
    fn storage_type_native() -> StorageType {
        FloatKind::E4M3.into()
    }
}
impl CubePrimitive for e4m3 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        Float8E4M3Type::get(scope.ctx()).into()
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e4m3::from_f64(value)
    }
}

impl IntoRuntime for e4m3 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for e4m3 {
    type Expand = NativeExpand<e4m3>;
    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for e4m3 {}

impl CubeType for e5m2 {
    type ExpandType = NativeExpand<e5m2>;
}

impl CubeDebug for e5m2 {}
impl Scalar for e5m2 {
    fn storage_type_native() -> StorageType {
        FloatKind::E5M2.into()
    }
}
impl CubePrimitive for e5m2 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        Float8E5M2Type::get(scope.ctx()).into()
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e5m2::from_f64(value)
    }
}

impl IntoRuntime for e5m2 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for e5m2 {
    type Expand = NativeExpand<e5m2>;
    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for e5m2 {}

impl CubeType for ue8m0 {
    type ExpandType = NativeExpand<ue8m0>;
}

impl CubeDebug for ue8m0 {}
impl Scalar for ue8m0 {
    fn storage_type_native() -> StorageType {
        FloatKind::UE8M0.into()
    }
}
impl CubePrimitive for ue8m0 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        Float8E8M0Type::get(scope.ctx()).into()
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        ue8m0::from_f64(value)
    }
}

impl IntoRuntime for ue8m0 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for ue8m0 {
    type Expand = NativeExpand<ue8m0>;
    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for ue8m0 {}
