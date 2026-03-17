use cubecl_common::{e2m1, e2m1x2};
use cubecl_ir::{ConstantValue, ElemType, FloatKind, Scope, StorageType, Type};

use crate::prelude::*;

impl CubeType for e2m1 {
    type ExpandType = NativeExpand<e2m1>;
}

impl Scalar for e2m1 {}
impl CubePrimitive for e2m1 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(StorageType::Scalar(ElemType::Float(FloatKind::E2M1)).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        e2m1::from_f64(value)
    }
}

impl IntoRuntime for e2m1 {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for e2m1 {}

impl CubeType for e2m1x2 {
    type ExpandType = NativeExpand<e2m1x2>;
}

// Considered a scalar because it's really just a `u8` in a trenchcoat, and should be possible to
// store in a `Vector`.
impl Scalar for e2m1x2 {}
impl CubePrimitive for e2m1x2 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2).into())
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
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for e2m1x2 {}
