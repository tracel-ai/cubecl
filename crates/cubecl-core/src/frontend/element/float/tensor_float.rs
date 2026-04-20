use cubecl_common::tf32;
use cubecl_ir::{ConstantValue, ElemType, FloatKind, Scope, Type};
use half::f16;

use crate::prelude::*;

use super::{CubePrimitive, CubeType, Float, IntoRuntime, NativeAssign, NativeExpand};

impl CubeType for tf32 {
    type ExpandType = NativeExpand<tf32>;
}

impl CubeDebug for tf32 {}
impl Scalar for tf32 {}
impl CubePrimitive for tf32 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type_native() -> Option<Type> {
        Some(ElemType::Float(FloatKind::TF32).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Float(value) = value else {
            unreachable!()
        };
        tf32::from_f64(value)
    }
}

impl IntoRuntime for tf32 {
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}
impl IntoExpand for tf32 {
    type Expand = NativeExpand<tf32>;
    fn into_expand(self, _: &Scope) -> Self::Expand {
        self.into()
    }
}

impl Numeric for tf32 {
    fn min_value() -> Self {
        Self::from_f32(f32::MIN)
    }
    fn max_value() -> Self {
        Self::from_f32(f32::MAX)
    }
}

impl NativeAssign for tf32 {}

impl Float for tf32 {
    const DIGITS: u32 = 32;

    const EPSILON: Self = tf32::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = tf32::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = 10;

    /// Maximum possible [`cubecl_common::tf32`] power of 10 exponent
    const MAX_10_EXP: i32 = 38;
    /// Maximum possible [`cubecl_common::tf32`] power of 2 exponent
    const MAX_EXP: i32 = 128;

    /// Minimum possible normal [`cubecl_common::tf32`] power of 10 exponent
    const MIN_10_EXP: i32 = -37;
    /// One greater than the minimum possible normal [`cubecl_common::tf32`] power of 2 exponent
    const MIN_EXP: i32 = -125;

    /// `MIN_POSITIVE` is defined by precision, so use `f16` as reference
    const MIN_POSITIVE: Self = tf32::from_f32(f16::MIN_POSITIVE.to_f32_const());

    const NAN: Self = tf32::from_f32(f32::NAN);

    const NEG_INFINITY: Self = tf32::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        tf32::from_f32(val)
    }
}
