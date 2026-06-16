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
    vector::DotOp,
};
use half::{bf16, f16};

pub mod sub {
    use cubecl_ir::{ConstantValue, ExpandValue, dialect::math::SubOp};

    use super::*;

    pub fn expand<C: CubePrimitive>(
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
            _ => binary_expand(scope, lhs.into(), rhs.into(), SubOp::new).into(),
        }
    }
}

pub mod clamp {
    use super::*;
    use cubecl_ir::{dialect::cmp::ClampOp, pliron::builtin::op_interfaces::OneResultInterface};

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        input: NativeExpand<C>,
        min: NativeExpand<C>,
        max: NativeExpand<C>,
    ) -> NativeExpand<C> {
        let input = input.read_value(scope);
        let min = min.read_value(scope);
        let max = max.read_value(scope);
        let op = ClampOp::new(scope.ctx_mut(), input, min, max);
        scope.register(&op);
        op.get_result(scope.ctx()).into()
    }
}

pub mod clamp_max {
    use super::*;
    use cubecl_ir::dialect::cmp::MinOp;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), MinOp::new).into()
    }
}

pub mod clamp_min {
    use super::*;
    use cubecl_ir::dialect::cmp::MaxOp;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), MaxOp::new).into()
    }
}

/// The minimum of two values, not requiring `Ord`. Provided for clarity in certain cases, though
/// `clamp_max` may sometimes be more clear.
pub fn min<T: PartialOrd + CubePrimitive>(lhs: T, rhs: T) -> T {
    clamp_max(lhs, rhs)
}

pub mod min {
    use super::*;
    use cubecl_ir::dialect::cmp::MinOp;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), MinOp::new).into()
    }
}

/// The maximum of two values, not requiring `Ord`. Provided for clarity in certain cases, though
/// `clamp_min` may sometimes be more clear.
pub fn max<T: PartialOrd + CubePrimitive>(lhs: T, rhs: T) -> T {
    clamp_min(lhs, rhs)
}

pub mod max {
    use super::*;
    use cubecl_ir::dialect::cmp::MaxOp;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), MaxOp::new).into()
    }
}

/// For binary functions without special syntax
macro_rules! impl_binary_func {
    ($trait_name:ident, $method_name:ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive + CubeType<ExpandType: [<$trait_name Expand>]> + Sized {
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

            $(impl $trait_name for $type {})*
            impl<T: CubePrimitive + $trait_name> [<$trait_name Expand>] for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Self) -> Self {
                    binary_expand(scope, self.into(), rhs.into(), $operator::new).into()
                }
            }
        }
    }
}

macro_rules! impl_binary_func_scalar_out {
    ($trait_name:ident, $method_name:ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive
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

            $(impl $trait_name for $type {})*
            impl<T: CubePrimitive + $trait_name> [<$trait_name Expand>] for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Self) -> Self::Scalar {
                    binary_expand(scope, self.into(), rhs.into(), $operator::new).into()
                }
            }
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

macro_rules! impl_core_binop {
    ($trait: ident, $method: ident, $op: expr) => {
        paste::paste! {
            pub trait [<Cube $trait>]: $trait<Output = Self> + CubePrimitive + IntoRuntime + CubeType<ExpandType: [<$trait Expand>]> + Sized {
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

            impl<T: $trait<Output = T> + CubePrimitive + IntoRuntime> [<Cube $trait>] for T {}
            impl<T: $trait<Output = T> + CubePrimitive> [<$trait Expand>] for NativeExpand<T> {
                fn [<__expand_ $method _method>](self, scope: &Scope, rhs: Self) -> Self {
                    binary_expand(scope, self.into(), rhs.into(), $op::new).into()
                }
            }
        }
    };
}

macro_rules! impl_core_assign_binop {
    ($trait: ident, $method: ident, $op: expr) => {
        paste::paste! {
            pub trait [<Cube $trait>]: $trait + CubePrimitive + CubeType<ExpandType: [<$trait Expand>]> + Sized {
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

            impl<T: $trait + CubePrimitive> [<Cube $trait>] for T {}
            impl<T: $trait + CubePrimitive> [<$trait Expand>] for NativeExpand<T> {
                fn [<__expand_ $method _method>](&mut self, scope: &Scope, rhs: Self) {
                    assign_binop_expand(scope, self, rhs, $op::new);
                }
            }
        }
    };
}

impl_core_binop!(Add, add, AddOp);
impl_core_binop!(Sub, sub, SubOp);
impl_core_binop!(Mul, mul, MulOp);
impl_core_binop!(Div, div, DivOp);
impl_core_binop!(Rem, rem, RemOp);

impl_core_assign_binop!(AddAssign, add_assign, AddOp);
impl_core_assign_binop!(SubAssign, sub_assign, SubOp);
impl_core_assign_binop!(MulAssign, mul_assign, MulOp);
impl_core_assign_binop!(DivAssign, div_assign, DivOp);
impl_core_assign_binop!(RemAssign, rem_assign, RemOp);

impl_core_binop!(BitAnd, bitand, BitwiseAndOp);
impl_core_binop!(BitOr, bitor, BitwiseOrOp);
impl_core_binop!(BitXor, bitxor, BitwiseXorOp);
impl_core_binop!(Shl, shl, ShiftLeftOp);
impl_core_binop!(Shr, shr, ShiftRightOp);

impl_core_assign_binop!(BitAndAssign, bitand_assign, BitwiseAndOp);
impl_core_assign_binop!(BitOrAssign, bitor_assign, BitwiseOrOp);
impl_core_assign_binop!(BitXorAssign, bitxor_assign, BitwiseXorOp);
impl_core_assign_binop!(ShlAssign, shl_assign, ShiftLeftOp);
impl_core_assign_binop!(ShrAssign, shr_assign, ShiftRightOp);

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

impl_binary_func!(Powf, powf, PowfOp, f16, bf16, flex32, tf32, f32, f64);

impl_binary_func!(Hypot, hypot, HypotOp, f16, bf16, flex32, tf32, f32, f64);

impl_binary_func!(Rhypot, rhypot, RhypotOp, f16, bf16, flex32, tf32, f32, f64);

impl_binary_func!(ArcTan2, atan2, ArcTan2Op, f16, bf16, flex32, tf32, f32, f64);
impl_binary_func!(
    ModFloor, mod_floor, ModFloorOp, f16, bf16, flex32, tf32, f32, f64, i8, i16, i32, i64, u8, u16,
    u32, u64, usize, isize
);
impl_binary_func!(MulHi, mul_hi, MulHiOp, i32, u32, usize, isize);
impl_binary_func!(
    SaturatingAdd,
    saturating_add,
    SaturatingAddOp,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    usize,
    isize
);
impl_binary_func!(
    SaturatingSub,
    saturating_sub,
    SaturatingSubOp,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    usize,
    isize
);
impl_binary_func_scalar_out!(
    Dot, dot, DotOp, f16, bf16, flex32, tf32, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
    usize, isize
);

impl_binary_func_mixed_types!(
    Powi, powi, i32, PowiOp, f16, bf16, flex32, tf32, f32, f64, i8, i16, i32, i64, u8, u16, u32,
    u64, usize, isize
);
