use cubecl_common::flex32;
use cubecl_ir::{ConstantValue, ElemType, FloatKind, Scope, Type};

use crate::prelude::*;

use super::{CubePrimitive, CubeType, Float, IntoRuntime, NativeAssign, NativeExpand};

impl CubeType for flex32 {
    type ExpandType = NativeExpand<flex32>;
}

impl CubeDebug for flex32 {}
impl Scalar for flex32 {}
impl CubePrimitive for flex32 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::Flex32).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        flex32::from_f64(value)
    }
}

impl IntoRuntime for flex32 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for flex32 {
    type Expand = NativeExpand<flex32>;
    fn into_expand(self, _: &Scope) -> Self::Expand {
        self.into()
    }
}

impl Numeric for flex32 {
    fn min_value() -> Self {
        <Self as num_traits::Float>::min_value()
    }
    fn max_value() -> Self {
        <Self as num_traits::Float>::max_value()
    }
}

impl NativeAssign for flex32 {}

impl Float for flex32 {
    const DIGITS: u32 = 32;

    const EPSILON: Self = flex32::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = flex32::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    /// Maximum possible [`flex32`] power of 10 exponent
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    /// Maximum possible [`flex32`] power of 2 exponent
    const MAX_EXP: i32 = f32::MAX_EXP;

    /// Minimum possible normal [`flex32`] power of 10 exponent
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;
    /// One greater than the minimum possible normal [`flex32`] power of 2 exponent
    const MIN_EXP: i32 = f32::MIN_EXP;

    const MIN_POSITIVE: Self = flex32::from_f32(f32::MIN_POSITIVE);

    const NAN: Self = flex32::from_f32(f32::NAN);

    const NEG_INFINITY: Self = flex32::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        flex32::from_f32(val)
    }
}
