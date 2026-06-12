use cubecl_common::{e2m3, e3m2};
use cubecl_ir::{
    ConstantValue, FloatKind, Scope,
    pliron::{context::Ptr, r#type::TypeObj},
    types::scalar::{ FloatType},
};

use crate::prelude::*;

impl CubeType for e2m3 {
    type ExpandType = NativeExpand<e2m3>;
}

impl CubeDebug for e2m3 {}
impl Scalar for e2m3 {
    fn storage_type_native() -> StorageType {
        FloatKind::E2M3.into()
    }
}
impl CubePrimitive for e2m3 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn __expand_as_type(scope: &Scope) -> Ptr<TypeObj> {
        FloatType::get(&mut scope.ctx_mut(), FloatKind::E2M3).into()
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("e2m3 doesn't yet support conversion");
    }
}

impl IntoRuntime for e2m3 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for e2m3 {
    type Expand = NativeExpand<e2m3>;
    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for e2m3 {}

impl CubeType for e3m2 {
    type ExpandType = NativeExpand<e3m2>;
}

impl CubeDebug for e3m2 {}
impl Scalar for e3m2 {
    fn storage_type_native() -> StorageType {
        FloatKind::E3M2.into()
    }
}
impl CubePrimitive for e3m2 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn __expand_as_type(scope: &Scope) -> Ptr<TypeObj> {
        FloatType::get(&mut scope.ctx_mut(), FloatKind::E3M2).into()
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("e3m2 doesn't yet support conversion");
    }
}

impl IntoRuntime for e3m2 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for e3m2 {
    type Expand = NativeExpand<e3m2>;
    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for e3m2 {}
