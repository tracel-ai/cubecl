use crate::ir::{ExpandValue, Scope};
use crate::{
    flex32,
    frontend::{CubePrimitive, NativeExpand},
    prelude::*,
};
use crate::{frontend::CubeType, tf32};
use crate::{frontend::operation::base::binary_expand, unexpanded};
use core::ops::*;
use cubecl_ir::dialect::{
    bitwise::*,
    general::{BoolAndOp, BoolOrOp},
    math::*,
    vector::{FDotOp, SDotOp, UDotOp},
};
use half::{bf16, f16};

pub mod sub {
    use cubecl_ir::{ConstantValue, ExpandValue};

    use super::*;

    pub fn expand<C: CubeSub>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        // Dirty hack to enable slice destructuring with trailing patterns on `Sequence`
        match (lhs.expand, rhs.expand.as_const()) {
            (
                ExpandValue::Constant {
                    value: ConstantValue::UInt(lhs_val),
                    ty,
                },
                Some(ConstantValue::UInt(rhs_val)),
            ) => {
                let value = (lhs_val - rhs_val).into();
                ExpandValue::constant(value, ty).into()
            }
            _ => C::Scalar::__expand_native_sub(scope, lhs.into(), rhs.into()).into(),
        }
    }
}

pub mod clamp {
    use super::*;

    pub fn expand<C: CubePartialOrd>(
        scope: &Scope,
        input: NativeExpand<C>,
        min: NativeExpand<C>,
        max: NativeExpand<C>,
    ) -> NativeExpand<C> {
        C::Scalar::__expand_native_clamp(scope, input.into(), min.into(), max.into()).into()
    }
}

pub mod clamp_max {
    use super::*;

    pub fn expand<C: CubePartialOrd>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        C::Scalar::__expand_native_min(scope, lhs.into(), rhs.into()).into()
    }
}

pub mod clamp_min {
    use super::*;

    pub fn expand<C: CubePartialOrd>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        C::Scalar::__expand_native_max(scope, lhs.into(), rhs.into()).into()
    }
}

/// The minimum of two values, not requiring `Ord`. Provided for clarity in certain cases, though
/// `clamp_max` may sometimes be more clear.
pub fn min<T: CubePartialOrd>(lhs: T, rhs: T) -> T {
    clamp_max(lhs, rhs)
}

pub mod min {
    use super::*;

    pub fn expand<C: CubePartialOrd>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        C::Scalar::__expand_native_min(scope, lhs.into(), rhs.into()).into()
    }
}

/// The maximum of two values, not requiring `Ord`. Provided for clarity in certain cases, though
/// `clamp_min` may sometimes be more clear.
pub fn max<T: CubePartialOrd>(lhs: T, rhs: T) -> T {
    clamp_min(lhs, rhs)
}

pub mod max {
    use super::*;

    pub fn expand<C: CubePartialOrd>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        C::Scalar::__expand_native_max(scope, lhs.into(), rhs.into()).into()
    }
}

/// For binary functions without special syntax
macro_rules! define_binary_func {
    ($trait_name:ident, $method_name:ident) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive<Scalar: [<$trait_name NativeExpand>]>
                + CubeType<ExpandType: [<$trait_name Expand>]> + Sized {
                fn $method_name(self, _rhs: Self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &Scope,
                    lhs: NativeExpand<Self>,
                    rhs: NativeExpand<Self>,
                ) -> NativeExpand<Self> {
                    lhs.[<__expand_ $method_name _method>](scope, rhs)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Self) -> Self;
            }

            pub trait [<$trait_name NativeExpand>] {
                fn [<__expand_native_ $method_name>](scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
            }

            impl<T: $trait_name> [<$trait_name Expand>] for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Self) -> Self {
                    T::Scalar::[<__expand_native_ $method_name>](scope, self.into(), rhs.into()).into()
                }
            }
        }
    }
}

macro_rules! impl_binary_func {
    ($($type:ty),*; $trait_name:ident, $method_name:ident, $operator:expr) => {
        paste::paste! {
            $(impl $trait_name for $type {})*
            $(impl [<$trait_name NativeExpand>] for $type {
                fn [<__expand_native_ $method_name>](scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue {
                    binary_expand(scope, lhs, rhs, $operator::new)
                }
            })*
        }
    }
}

macro_rules! define_binary_func_scalar_out {
    ($trait_name:ident, $method_name:ident) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive<Scalar: [<$trait_name NativeExpand>]>
                + CubeType<ExpandType: [<$trait_name Expand>]
                + CubePrimitiveExpand<Scalar = NativeExpand<Self::Scalar>>>
                + Sized {
                fn $method_name(self, _rhs: Self) -> Self::Scalar {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &Scope,
                    lhs: NativeExpand<Self>,
                    rhs: NativeExpand<Self>,
                ) -> NativeExpand<Self::Scalar> {
                    lhs.[<__expand_ $method_name _method>](scope, rhs)
                }
            }

            pub trait [<$trait_name Expand>]: CubePrimitiveExpand {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Self) -> Self::Scalar;
            }

            pub trait [<$trait_name NativeExpand>] {
                fn [<__expand_native_ $method_name _scalar>](scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
                fn [<__expand_native_ $method_name>](scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
            }

            impl<T: CubePrimitive + $trait_name> [<$trait_name Expand>] for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Self) -> Self::Scalar {
                    // A lot of backends can't deal with 1-sized vectors, and we want to validate
                    // that the input is a vector
                    if self.__expand_vector_size_method(scope) == 1 {
                        T::Scalar::[<__expand_native_ $method_name _scalar>](scope, self.into(), rhs.into()).into()
                    } else {
                        T::Scalar::[<__expand_native_ $method_name>](scope, self.into(), rhs.into()).into()
                    }
                }
            }
        }
    }
}

macro_rules! impl_binary_func_scalar_out {
    ($($type:ty),*; $trait_name:ident, $method_name:ident, $operator:expr, $scalar_op:expr) => {
        paste::paste! {
            $(impl $trait_name for $type {})*
            $(impl [<$trait_name NativeExpand>] for $type {
                fn [<__expand_native_ $method_name _scalar>](scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue {
                    binary_expand(scope, lhs, rhs, $scalar_op::new)
                }
                fn [<__expand_native_ $method_name>](scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue {
                    binary_expand(scope, lhs, rhs, $operator::new)
                }
            })*
        }
    }
}

macro_rules! impl_binary_func_mixed_types {
    ($trait_name:ident, $method_name:ident, $rhs_ty: ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name<Rhs: CubePrimitive + CubeType<ExpandType: Into<ExpandValue>> + Sized>:
                CubePrimitive + CubeType<ExpandType: [<$trait_name Expand>]<Rhs>> + Sized {
                fn $method_name(self, _rhs: Rhs) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &Scope,
                    lhs: NativeExpand<Self>,
                    rhs: NativeExpand<Rhs>,
                ) -> NativeExpand<Self> {
                    binary_expand(scope, lhs.into(), rhs.into(), $operator::new).into()
                }
            }

            pub trait [<$trait_name Expand>]<Rhs: CubeType>{
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Rhs::ExpandType) -> Self;
            }

            $(impl $trait_name<$rhs_ty> for $type {})*
            impl<Rhs: CubePrimitive, T: CubePrimitive + $trait_name<Rhs>> [<$trait_name Expand>]<Rhs> for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: NativeExpand<Rhs>) -> Self {
                    binary_expand(scope, self.into(), rhs.into(), $operator::new).into()
                }
            }
        }
    }
}

macro_rules! define_core_binop {
    ($trait: ident, $method: ident) => {
        paste::paste! {
            pub trait [<Cube $trait>]:
                $trait<Output = Self> + CubePrimitive<Scalar: [<$trait NativeExpand>]> + IntoRuntime
                + CubeType<ExpandType: [<$trait Expand>]> + Sized {
                fn [<__expand_ $method _method>](self, scope: &Scope, rhs: NativeExpand<Self>) -> NativeExpand<Self> {
                    let this = self.__expand_runtime_method(scope);
                    this.[<__expand_ $method _method>](scope, rhs)
                }

                fn [<__expand_ $method>](
                    scope: &Scope,
                    lhs: NativeExpand<Self>,
                    rhs: NativeExpand<Self>,
                ) -> NativeExpand<Self> {
                    lhs.[<__expand_ $method _method>](scope, rhs)
                }
            }

            pub trait [<$trait Expand>] {
                fn [<__expand_ $method _method>](self, scope: &Scope, rhs: Self) -> Self;
            }

            pub trait [<$trait NativeExpand>] {
                fn [<__expand_native_ $method>](scope: &Scope, this: ExpandValue, rhs: ExpandValue) -> ExpandValue;
            }

            impl<T: $trait<Output = Self> + CubePrimitive<Scalar: [<$trait NativeExpand>]> + IntoRuntime> [<Cube $trait>] for T {}
            impl<T: [<Cube $trait>]> [<$trait Expand>] for NativeExpand<T> {
                fn [<__expand_ $method _method>](self, scope: &Scope, rhs: Self) -> Self {
                    T::Scalar::[<__expand_native_ $method>](scope, self.into(), rhs.into()).into()
                }
            }
        }
    };
}

macro_rules! impl_core_binop {
    ($($ty: ty),*; $trait: ident, $method: ident, $op: expr) => {
        paste::paste! {
            $(impl [<$trait NativeExpand>] for $ty {
                fn [<__expand_native_ $method>](scope: &Scope, this: ExpandValue, rhs: ExpandValue) -> ExpandValue {
                    binary_expand(scope, this, rhs, $op::new)
                }
            })*
        }
    };
}
macro_rules! define_core_assign_binop {
    ($trait: ident, $base_trait: ident, $method: ident, $base_method: ident) => {
        paste::paste! {
            pub trait [<Cube $trait>]: $trait + CubePrimitive<Scalar: [<$base_trait NativeExpand>]>
                + CubeType<ExpandType: [<$trait Expand>]> + Sized {
                fn [<__expand_ $method>](
                    scope: &Scope,
                    lhs: &mut NativeExpand<Self>,
                    rhs: NativeExpand<Self>,
                ) {
                    lhs.[<__expand_ $method _method>](scope, rhs)
                }
            }

            pub trait [<$trait Expand>] {
                fn [<__expand_ $method _method>](&mut self, scope: &Scope, rhs: Self);
            }

            impl<T: $trait + [<Cube $base_trait>]> [<Cube $trait>] for T {}
            impl<T: $trait + [<Cube $base_trait>]> [<$trait Expand>] for NativeExpand<T> {
                fn [<__expand_ $method _method>](&mut self, scope: &Scope, rhs: Self) {
                    assign_binop_expand(scope, self, rhs, T::Scalar::[<__expand_native_ $base_method>]);
                }
            }
        }
    };
}

define_core_binop!(Add, add);
define_core_binop!(Sub, sub);
define_core_binop!(Mul, mul);
define_core_binop!(Div, div);
define_core_binop!(Rem, rem);

impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; Add, add, IAddOp);
impl_core_binop!(f16, bf16, f32, flex32, tf32, f64; Add, add, FAddOp);

impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; Sub, sub, ISubOp);
impl_core_binop!(f16, bf16, f32, flex32, tf32, f64; Sub, sub, FSubOp);

impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; Mul, mul, IMulOp);
impl_core_binop!(f16, bf16, f32, flex32, tf32, f64; Mul, mul, FMulOp);

impl_core_binop!(i8, i16, i32, i64, isize; Div, div, SDivOp);
impl_core_binop!(u8, u16, u32, u64, usize; Div, div, UDivOp);
impl_core_binop!(f16, bf16, f32, flex32, tf32, f64; Div, div, FDivOp);

impl_core_binop!(i8, i16, i32, i64, isize; Rem, rem, SRemOp);
impl_core_binop!(u8, u16, u32, u64, usize; Rem, rem, URemOp);
impl_core_binop!(f16, bf16, f32, flex32, tf32, f64; Rem, rem, FRemOp);

define_core_assign_binop!(AddAssign, Add, add_assign, add);
define_core_assign_binop!(SubAssign, Sub, sub_assign, sub);
define_core_assign_binop!(MulAssign, Mul, mul_assign, mul);
define_core_assign_binop!(DivAssign, Div, div_assign, div);
define_core_assign_binop!(RemAssign, Rem, rem_assign, rem);

define_core_binop!(BitAnd, bitand);
define_core_binop!(BitOr, bitor);
define_core_binop!(BitXor, bitxor);
define_core_binop!(Shl, shl);
define_core_binop!(Shr, shr);

impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; BitAnd, bitand, BitwiseAndOp);
impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; BitOr, bitor, BitwiseOrOp);
impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; BitXor, bitxor, BitwiseXorOp);
impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; Shl, shl, ShiftLeftOp);
impl_core_binop!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; Shr, shr, ShiftRightOp);

define_core_assign_binop!(BitAndAssign, BitAnd, bitand_assign, bitand);
define_core_assign_binop!(BitOrAssign, BitOr, bitor_assign, bitor);
define_core_assign_binop!(BitXorAssign, BitXor, bitxor_assign, bitxor);
define_core_assign_binop!(ShlAssign, Shl, shl_assign, shl);
define_core_assign_binop!(ShrAssign, Shr, shr_assign, shr);

impl BitAndNativeExpand for bool {
    fn __expand_native_bitand(scope: &Scope, this: ExpandValue, rhs: ExpandValue) -> ExpandValue {
        binary_expand(scope, this, rhs, BoolAndOp::new)
    }
}
impl BitOrNativeExpand for bool {
    fn __expand_native_bitor(scope: &Scope, this: ExpandValue, rhs: ExpandValue) -> ExpandValue {
        binary_expand(scope, this, rhs, BoolOrOp::new)
    }
}

pub trait CubeAnd:
    CubePrimitive + Into<ExpandValue> + CubeType<ExpandType: AndExpand> + Sized
{
    fn __expand_and_method(self, scope: &Scope, rhs: NativeExpand<Self>) -> NativeExpand<Self> {
        let this: ExpandValue = self.into();
        let this: NativeExpand<Self> = this.into();
        this.__expand_and_method(scope, rhs)
    }
    fn __expand_and(
        scope: &Scope,
        lhs: NativeExpand<Self>,
        rhs: NativeExpand<Self>,
    ) -> NativeExpand<Self> {
        lhs.__expand_and_method(scope, rhs)
    }
}
pub trait AndExpand {
    fn __expand_and_method(self, scope: &Scope, rhs: Self) -> Self;
}

impl CubeAnd for bool {}
impl<T: CubeAnd + CubePrimitive> AndExpand for NativeExpand<T> {
    fn __expand_and_method(self, scope: &Scope, rhs: Self) -> Self {
        binary_expand(scope, self.into(), rhs.into(), BoolAndOp::new).into()
    }
}

pub trait CubeOr:
    CubePrimitive + Into<ExpandValue> + CubeType<ExpandType: OrExpand> + Sized
{
    fn __expand_or_method(self, scope: &Scope, rhs: NativeExpand<Self>) -> NativeExpand<Self> {
        let this: ExpandValue = self.into();
        let this: NativeExpand<Self> = this.into();
        this.__expand_or_method(scope, rhs)
    }
    fn __expand_or(
        scope: &Scope,
        lhs: NativeExpand<Self>,
        rhs: NativeExpand<Self>,
    ) -> NativeExpand<Self> {
        lhs.__expand_or_method(scope, rhs)
    }
}
pub trait OrExpand {
    fn __expand_or_method(self, scope: &Scope, rhs: Self) -> Self;
}

impl CubeOr for bool {}
impl<T: CubeOr + CubePrimitive> OrExpand for NativeExpand<T> {
    fn __expand_or_method(self, scope: &Scope, rhs: Self) -> Self {
        binary_expand(scope, self.into(), rhs.into(), BoolOrOp::new).into()
    }
}

define_binary_func!(Powf, powf);
impl_binary_func!(f16, bf16, flex32, tf32, f32, f64; Powf, powf, PowfOp);

define_binary_func!(Hypot, hypot);
impl_binary_func!(f16, bf16, flex32, tf32, f32, f64; Hypot, hypot, HypotOp);

define_binary_func!(Rhypot, rhypot);
impl_binary_func!(f16, bf16, flex32, tf32, f32, f64; Rhypot, rhypot, RhypotOp);

define_binary_func!(ArcTan2, atan2);
impl_binary_func!(f16, bf16, flex32, tf32, f32, f64; ArcTan2, atan2, ArcTan2Op);

define_binary_func!(ModFloor, mod_floor);
impl_binary_func!(i8, i16, i32, i64, isize; ModFloor, mod_floor, SModFloorOp);
impl_binary_func!(u8, u16, u32, u64, usize; ModFloor, mod_floor, URemOp);
impl_binary_func!(f16, bf16, flex32, tf32, f32, f64; ModFloor, mod_floor, FModFloorOp);

define_binary_func!(MulHi, mul_hi);
impl_binary_func!(i32, i64, isize; MulHi, mul_hi, SMulHiOp);
impl_binary_func!(u32, u64, usize; MulHi, mul_hi, UMulHiOp);

define_binary_func!(SaturatingAdd, saturating_add);
impl_binary_func!(i8, i16, i32, i64, isize; SaturatingAdd, saturating_add, SaturatingSAddOp);
impl_binary_func!(u8, u16, u32, u64, usize; SaturatingAdd, saturating_add, SaturatingUAddOp);

define_binary_func!(SaturatingSub, saturating_sub);
impl_binary_func!(i8, i16, i32, i64, isize; SaturatingSub, saturating_sub, SaturatingSSubOp);
impl_binary_func!(u8, u16, u32, u64, usize; SaturatingSub, saturating_sub, SaturatingUSubOp);

define_binary_func_scalar_out!(Dot, dot);
impl_binary_func_scalar_out!(i8, i16, i32, i64, isize; Dot, dot, SDotOp, IMulOp);
impl_binary_func_scalar_out!(u8, u16, u32, u64, usize; Dot, dot, UDotOp, IMulOp);
impl_binary_func_scalar_out!(f16, bf16, flex32, tf32, f32, f64; Dot, dot, FDotOp, FMulOp);

impl_binary_func_mixed_types!(
    Powi, powi, i32, PowiOp, f16, bf16, flex32, tf32, f32, f64, i8, i16, i32, i64, u8, u16, u32,
    u64, usize, isize
);
