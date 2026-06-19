use cubecl_common::{e4m3, e5m2, ue8m0};
use cubecl_ir::{ConstantValue, ElemType, FloatKind, Scope, Type};

use crate::prelude::*;

impl CubeType for e4m3 {
    type ExpandType = NativeExpand<e4m3>;
}

impl CubeDebug for e4m3 {}
impl Scalar for e4m3 {}
impl CubePrimitive for e4m3 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

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

impl Numeric for e4m3 {
    fn min_value() -> Self {
        Self::MIN
    }

    fn max_value() -> Self {
        Self::MAX
    }
}

impl Numeric for e5m2 {
    fn min_value() -> Self {
        Self::MIN
    }

    fn max_value() -> Self {
        Self::MAX
    }
}

impl Float for e4m3 {
    const DIGITS: u32 = e4m3::DIGITS;
    const EPSILON: Self = e4m3::EPSILON;
    const INFINITY: Self = e4m3::INFINITY;
    const MANTISSA_DIGITS: u32 = e4m3::MANTISSA_DIGITS;
    const MAX_10_EXP: i32 = e4m3::MAX_10_EXP;
    const MAX_EXP: i32 = e4m3::MAX_EXP;
    const MIN_10_EXP: i32 = e4m3::MIN_10_EXP;
    const MIN_EXP: i32 = e4m3::MIN_EXP;
    const MIN_POSITIVE: Self = e4m3::MIN_POSITIVE;
    const NAN: Self = e4m3::NAN;
    const NEG_INFINITY: Self = e4m3::NEG_INFINITY;
    const RADIX: u32 = e4m3::RADIX;

    fn new(val: f32) -> Self {
        Self::from_f32(val)
    }
}

impl CubeType for e5m2 {
    type ExpandType = NativeExpand<e5m2>;
}

impl CubeDebug for e5m2 {}
impl Scalar for e5m2 {}
impl CubePrimitive for e5m2 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

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
impl Float for e5m2 {
    const DIGITS: u32 = e5m2::DIGITS;
    const EPSILON: Self = e5m2::EPSILON;
    const INFINITY: Self = e5m2::INFINITY;
    const MANTISSA_DIGITS: u32 = e5m2::MANTISSA_DIGITS;
    const MAX_10_EXP: i32 = e5m2::MAX_10_EXP;
    const MAX_EXP: i32 = e5m2::MAX_EXP;
    const MIN_10_EXP: i32 = e5m2::MIN_10_EXP;
    const MIN_EXP: i32 = e5m2::MIN_EXP;
    const MIN_POSITIVE: Self = e5m2::MIN_POSITIVE;
    const NAN: Self = e5m2::NAN;
    const NEG_INFINITY: Self = e5m2::NEG_INFINITY;
    const RADIX: u32 = e5m2::RADIX;

    fn new(val: f32) -> Self {
        Self::from_f32(val)
    }
}
impl CubeType for ue8m0 {
    type ExpandType = NativeExpand<ue8m0>;
}

impl CubeDebug for ue8m0 {}
impl Scalar for ue8m0 {}
impl CubePrimitive for ue8m0 {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

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
