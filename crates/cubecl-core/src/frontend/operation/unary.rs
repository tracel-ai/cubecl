use core::ops::{Div, Neg, Not};
use cubecl_common::{e2m1, e2m1x2, e4m3, e5m2, ue8m0};
use cubecl_ir::dialect::{bitwise::*, general::BoolNotOp, math::*, vector::*};
use half::{bf16, f16};

use crate::{
    flex32,
    frontend::{DivExpand, Scalar},
    ir::{ExpandValue, Scope},
    prelude::{
        CubePrimitive, CubePrimitiveExpand, CubeType, IntoExpand, NativeExpand, Reinterpret,
    },
    tf32, unexpanded,
};

use super::base::unary_expand;

pub mod not {
    use super::*;

    pub fn expand<T: CubeNot>(scope: &Scope, x: NativeExpand<T>) -> NativeExpand<T> {
        if T::Scalar::elem_type(scope).is_bool() {
            unary_expand(scope, x.into(), BoolNotOp::new).into()
        } else {
            unary_expand(scope, x.into(), BitwiseNotOp::new).into()
        }
    }
}

macro_rules! define_unary_func {
    ($trait_name:ident, $method_name:ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive<Scalar: [<$trait_name NativeExpand>]>
                + CubeType<ExpandType: [<$trait_name Expand>]> + Sized {
                #[allow(unused_variables)]
                fn $method_name(self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](scope: &Scope, x: NativeExpand<Self>) -> NativeExpand<Self> {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self;
            }

            pub trait [<$trait_name NativeExpand>] {
                fn [<__expand_native_ $method_name>](scope: &Scope, input: ExpandValue) -> ExpandValue;
            }

            $(impl $trait_name for $type {})*
            $(impl [<$trait_name NativeExpand>] for $type {
                fn [<__expand_native_ $method_name>](scope: &Scope, input: ExpandValue) -> ExpandValue {
                    unary_expand(scope, input, $operator::new)
                }
            })*

            impl<T: $trait_name + CubePrimitive> [<$trait_name Expand>] for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self {
                    T::Scalar::[<__expand_native_ $method_name>](scope, self.into()).into()
                }
            }
        }
    }
}

macro_rules! impl_unary_func {
    ($($type:ty),*; $trait_name:ident, $method_name:ident, $operator:expr) => {
        paste::paste! {
            $(impl $trait_name for $type {})*
            $(impl [<$trait_name NativeExpand>] for $type {
                fn [<__expand_native_ $method_name>](scope: &Scope, input: ExpandValue) -> ExpandValue {
                    unary_expand(scope, input, $operator::new)
                }
            })*
        }
    }
}

macro_rules! impl_unary_func_nop {
    ($($type:ty),*; $trait_name:ident, $method_name:ident) => {
        paste::paste! {
            $(impl $trait_name for $type {})*
            $(impl [<$trait_name NativeExpand>] for $type {
                fn [<__expand_native_ $method_name>](_scope: &Scope, input: ExpandValue) -> ExpandValue {
                    input
                }
            })*
        }
    }
}

// Special handling for scalars
macro_rules! impl_normalize {
    ($trait_name:ident, $method_name:ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive + CubeType<ExpandType: [<$trait_name Expand>]> + Sized + Abs + Div<Output = Self> {
                #[allow(unused_variables)]
                fn $method_name(self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](scope: &Scope, x: NativeExpand<Self>) -> NativeExpand<Self> {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self;
            }

            $(impl $trait_name for $type {})*
            impl<T: $trait_name + CubePrimitive> [<$trait_name Expand>] for NativeExpand<T> where NativeExpand<T>: DivExpand {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self {
                    if self.__expand_vector_size_method(scope) == 1 {
                        // Sign might work, but dividing by `abs` preserves the NaN when normalizing 0.0
                        let abs = self.__expand_abs_method(scope);
                        self.__expand_div_method(scope, abs)
                    } else {
                        unary_expand(scope, self.into(), $operator::new).into()
                    }
                }
            }
        }
    }
}

macro_rules! define_unary_func_scalar_out {
    ($trait_name:ident, $method_name:ident) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive<Scalar: [<$trait_name NativeExpand>]>
                + CubeType<ExpandType: [<$trait_name Expand>]
                + CubePrimitiveExpand<Scalar = NativeExpand<Self::Scalar>>>
                + Sized {
                #[allow(unused_variables)]
                fn $method_name(self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](scope: &Scope, x: NativeExpand<Self>) -> NativeExpand<Self::Scalar> {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>]: CubePrimitiveExpand {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self::Scalar;
            }

            pub trait [<$trait_name NativeExpand>] {
                fn [<__expand_native_ $method_name _scalar>](scope: &Scope, input: ExpandValue) -> ExpandValue;
                fn [<__expand_native_ $method_name>](scope: &Scope, input: ExpandValue) -> ExpandValue;
            }

            impl<T: $trait_name + CubePrimitive> [<$trait_name Expand>] for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self::Scalar {
                    // A lot of backends can't deal with 1-sized vectors, and we don't want to deal
                    // with the scalar edge case.
                    if self.__expand_vector_size_method(scope) == 1 {
                        T::Scalar::[<__expand_native_ $method_name _scalar>](scope, self.into()).into()
                    } else {
                        T::Scalar::[<__expand_native_ $method_name>](scope, self.into()).into()
                    }
                }
            }
        }
    }
}

macro_rules! impl_unary_func_scalar_out {
    ($($type:ty),*; $trait_name:ident, $method_name:ident, $operator:expr, $scalar_op: expr) => {
        paste::paste! {
            $(impl $trait_name for $type {})*
            $(impl [<$trait_name NativeExpand>] for $type {
                fn [<__expand_native_ $method_name _scalar>](scope: &Scope, input: ExpandValue) -> ExpandValue {
                    ($scalar_op)(scope, input)
                }
                fn [<__expand_native_ $method_name>](scope: &Scope, input: ExpandValue) -> ExpandValue {
                    unary_expand(scope, input, $operator::new)
                }
            })*
        }
    }
}

macro_rules! impl_unary_func_fixed_out_ty {
    ($trait_name:ident, $method_name:ident, $out_ty: ty, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive + CubeType<ExpandType: [<$trait_name Expand>]
            + CubePrimitiveExpand<WithScalar<$out_ty> = NativeExpand<Self::WithScalar<$out_ty>>>> + Sized {
                #[allow(unused_variables, clippy::wrong_self_convention)]
                fn $method_name(self) -> Self::WithScalar<$out_ty> {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](scope: &Scope, x: NativeExpand<Self>) -> NativeExpand<Self::WithScalar<$out_ty>> {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>]: CubePrimitiveExpand {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self::WithScalar<$out_ty>;
            }

            $(impl $trait_name for $type {})*
            impl<T: $trait_name + CubePrimitive> [<$trait_name Expand>] for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self::WithScalar<$out_ty> {
                    unary_expand(scope, self.into(), $operator::new).into()
                }
            }
        }
    }
}

// Needs special handling because Rust combines bitwise and logical or into one trait
macro_rules! impl_not {
    ($trait_name:ident, $method_name:ident, $($type:ty),*) => {
        paste::paste! {
            pub trait [<Cube $trait_name>]:
                $trait_name<Output = Self>
                + CubePrimitive
                + CubeType<ExpandType: [<$trait_name Expand>]>
                + IntoExpand<Expand = <Self as CubeType>::ExpandType> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> NativeExpand<Self> {
                    let this = self.into_expand(scope);
                    this.[<__expand_ $method_name _method>](scope)
                }

                fn [<__expand_ $method_name>](scope: &Scope, x: NativeExpand<Self>) -> NativeExpand<Self> {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self;
            }

            $(impl [<Cube $trait_name>] for $type {})*
            impl<T: [<Cube $trait_name>] + CubePrimitive> [<$trait_name Expand>] for NativeExpand<T> {
                    fn [<__expand_ $method_name _method>](self, scope: &Scope) -> Self {
                    not::expand(scope, self.into())
                }
            }
        }
    }
}

macro_rules! define_core_unop {
    ($trait: ident, $method: ident) => {
        paste::paste! {
            pub trait [<Cube $trait>]:
                $trait<Output = Self> + CubePrimitive<Scalar: [<$trait NativeExpand>]>
                + IntoExpand<Expand = <Self as CubeType>::ExpandType>
                + CubeType<ExpandType: [<$trait Expand>]> + Sized {
                fn [<__expand_ $method _method>](self, scope: &Scope) -> NativeExpand<Self> {
                    let this: NativeExpand<Self> = self.into_expand(scope);
                    this.[<__expand_ $method _method>](scope)
                }

                fn [<__expand_ $method>](
                    scope: &Scope,
                    lhs: NativeExpand<Self>,
                ) -> NativeExpand<Self> {
                    lhs.[<__expand_ $method _method>](scope)
                }
            }

            pub trait [<$trait Expand>] {
                fn [<__expand_ $method _method>](self, scope: &Scope) -> Self;
            }

            pub trait [<$trait NativeExpand>] {
                fn [<__expand_native_ $method>](scope: &Scope, this: ExpandValue) -> ExpandValue;
            }

            impl<T: $trait<Output = Self> + CubePrimitive<Scalar: [<$trait NativeExpand>]>
                + IntoExpand<Expand = <Self as CubeType>::ExpandType>> [<Cube $trait>] for T {}
            impl<T: [<Cube $trait>]> [<$trait Expand>] for NativeExpand<T> {
                fn [<__expand_ $method _method>](self, scope: &Scope) -> Self {
                    T::Scalar::[<__expand_native_ $method>](scope, self.expand).into()
                }
            }
        }
    };
}

macro_rules! impl_core_unop {
    ($($ty: ty),*; $trait: ident, $method: ident, $op: expr) => {
        paste::paste! {
            $(impl [<$trait NativeExpand>] for $ty {
                fn [<__expand_native_ $method>](scope: &Scope, this: ExpandValue) -> ExpandValue {
                    unary_expand(scope, this, $op::new).into()
                }
            })*
        }
    };
}

impl_not!(
    Not, not, bool, u8, u16, u32, u64, i8, i16, i32, i64, isize, usize
);

define_core_unop!(Neg, neg);
impl_core_unop!(i8, i16, i32, i64, isize; Neg, neg, SNegOp);
impl_core_unop!(f16, bf16, f32, flex32, tf32, f64; Neg, neg, FNegOp);

define_unary_func!(Abs, abs, SAbsOp, i8, i16, i32, i64, isize);
impl_unary_func_nop!(u8, u16, u32, u64, usize; Abs, abs);
impl_unary_func!(e2m1, e4m3, e5m2, ue8m0, f16, bf16, flex32, tf32, f32, f64; Abs, abs, FAbsOp);

define_unary_func!(Exp, exp, ExpOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Log, ln, LogOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Log1p, log1p, Log1pOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Expm1, exp_m1, Expm1Op, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Cos, cos, CosOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Sin, sin, SinOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Tan, tan, TanOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Tanh, tanh, TanhOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Sinh, sinh, SinhOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Cosh, cosh, CoshOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(ArcCos, acos, ArcCosOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(ArcSin, asin, ArcSinOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(ArcTan, atan, ArcTanOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(ArcSinh, asinh, ArcSinhOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(ArcCosh, acosh, ArcCoshOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(ArcTanh, atanh, ArcTanhOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(
    Degrees, to_degrees, DegreesOp, f16, bf16, flex32, tf32, f32, f64
);
define_unary_func!(
    Radians, to_radians, RadiansOp, f16, bf16, flex32, tf32, f32, f64
);
define_unary_func!(Sqrt, sqrt, SqrtOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(
    InverseSqrt,
    inverse_sqrt,
    RsqrtOp,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
define_unary_func!(Round, round, RoundOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Floor, floor, FloorOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Ceil, ceil, CeilOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Trunc, trunc, TruncOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Erf, erf, ErfOp, f16, bf16, flex32, tf32, f32, f64);
define_unary_func!(Recip, recip, RecipOp, f16, bf16, flex32, tf32, f32, f64);

define_unary_func_scalar_out!(Magnitude, magnitude);
impl_unary_func_scalar_out!(f16, bf16, flex32, tf32, f32, f64; Magnitude, magnitude, MagnitudeOp, |scope, input| unary_expand(scope, input, FAbsOp::new));

define_unary_func_scalar_out!(VectorSum, vector_sum);
impl_unary_func_scalar_out!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; VectorSum, vector_sum, ISumOp, |_, input| input);
impl_unary_func_scalar_out!(f16, bf16, f32, flex32, tf32, f64; VectorSum, vector_sum, FSumOp, |_, input| input);

impl_normalize!(
    Normalize,
    normalize,
    NormalizeOp,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func_fixed_out_ty!(
    CountOnes,
    count_ones,
    u32,
    CountOnesOp,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    usize,
    isize
);
define_unary_func!(
    ReverseBits,
    reverse_bits,
    ReverseBitsOp,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    usize,
    isize
);

impl_unary_func_fixed_out_ty!(
    LeadingZeros,
    leading_zeros,
    u32,
    LeadingZerosBitsOp,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    usize,
    isize
);
impl_unary_func_fixed_out_ty!(
    TrailingZeros,
    trailing_zeros,
    u32,
    TrailingZerosBitsOp,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    usize,
    isize
);
impl_unary_func_fixed_out_ty!(
    FindFirstSet,
    find_first_set,
    u32,
    FindFirstSetOp,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    usize,
    isize
);
impl_unary_func_fixed_out_ty!(
    IsNan, is_nan, bool, IsNanOp, f16, bf16, flex32, tf32, f32, f64
);
impl_unary_func_fixed_out_ty!(
    IsInf, is_inf, bool, IsInfOp, f16, bf16, flex32, tf32, f32, f64
);

pub trait FloatBits:
    CubePrimitive + CubeType<ExpandType: FloatBitsExpand<Bits = Self::Bits>>
{
    type Bits: CubePrimitive;

    fn __expand_from_bits(scope: &Scope, bits: NativeExpand<Self::Bits>) -> NativeExpand<Self> {
        Self::__expand_reinterpret(scope, bits)
    }

    fn __expand_to_bits(scope: &Scope, this: NativeExpand<Self>) -> NativeExpand<Self::Bits> {
        <Self::Bits as Reinterpret>::__expand_reinterpret(scope, this)
    }
}

pub trait FloatBitsExpand: Sized {
    type Bits: CubePrimitive;

    fn __expand_to_bits_method(self, scope: &Scope) -> NativeExpand<Self::Bits>;
}

impl<F: FloatBits> FloatBitsExpand for NativeExpand<F> {
    type Bits = F::Bits;

    fn __expand_to_bits_method(self, scope: &Scope) -> NativeExpand<Self::Bits> {
        <Self::Bits as Reinterpret>::__expand_reinterpret(scope, self)
    }
}

impl FloatBits for e2m1x2 {
    type Bits = u8;
}

impl FloatBits for e5m2 {
    type Bits = u8;
}

impl FloatBits for e4m3 {
    type Bits = u8;
}

impl FloatBits for f16 {
    type Bits = u16;
}

impl FloatBits for bf16 {
    type Bits = u16;
}

impl FloatBits for f32 {
    type Bits = u32;
}

impl FloatBits for f64 {
    type Bits = u64;
}
