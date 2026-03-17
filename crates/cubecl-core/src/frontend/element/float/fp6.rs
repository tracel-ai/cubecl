use cubecl_common::{e2m3, e3m2};
use cubecl_ir::{ConstantValue, ElemType, FloatKind, Scope, Type};

use crate::prelude::*;

impl CubeType for e2m3 {
    type ExpandType = NativeExpand<e2m3>;
}

impl Scalar for e2m3 {}
impl CubePrimitive for e2m3 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::E2M3).into())
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("e2m3 doesn't yet support conversion");
    }
}

impl IntoRuntime for e2m3 {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for e2m3 {}

impl CubeType for e3m2 {
    type ExpandType = NativeExpand<e3m2>;
}

impl Scalar for e3m2 {}
impl CubePrimitive for e3m2 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::E3M2).into())
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("e3m2 doesn't yet support conversion");
    }
}

impl IntoRuntime for e3m2 {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for e3m2 {}
