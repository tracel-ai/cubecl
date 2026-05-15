use crate::ir::{Arithmetic, Bitwise, Scope, Variable};
use crate::{
    flex32,
    frontend::{CubePrimitive, NativeExpand},
    prelude::*,
};
use crate::{frontend::CubeType, tf32};
use crate::{
    frontend::operation::base::{binary_expand, binary_expand_fixed_output},
    unexpanded,
};
use core::ops::*;
use cubecl_ir::{ClampOperands, Operator};
use half::{bf16, f16};

pub mod sub {
    use cubecl_ir::{ConstantValue, Variable};

    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        // Dirty hack to enable slice destructuring with trailing patterns on `Sequence`
        match (lhs.expand.as_const(), rhs.expand.as_const()) {
            (Some(ConstantValue::UInt(lhs_val)), Some(ConstantValue::UInt(rhs_val))) => {
                let item_lhs = lhs.expand.value_type();
                let item_rhs = rhs.expand.value_type();

                let vector_size = find_vectorization(item_lhs, item_rhs);

                let item = item_lhs.with_vector_size(vector_size);
                let value = (lhs_val - rhs_val).into();
                Variable::constant(value, item).into()
            }
            _ => binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Sub).into(),
        }
    }
}

pub mod clamp {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        input: NativeExpand<C>,
        min: NativeExpand<C>,
        max: NativeExpand<C>,
    ) -> NativeExpand<C> {
        unary_expand(scope, input.into(), |op| {
            Arithmetic::Clamp(ClampOperands {
                input: op.input,
                min_value: min.expand,
                max_value: max.expand,
            })
        })
        .into()
    }
}

pub mod clamp_max {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Min).into()
    }
}

pub mod clamp_min {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Max).into()
    }
}

/// The minimum of two values, not requiring `Ord`. Provided for clarity in certain cases, though
/// `clamp_max` may sometimes be more clear.
pub fn min<T: PartialOrd + CubePrimitive>(lhs: T, rhs: T) -> T {
    clamp_max(lhs, rhs)
}

pub mod min {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Min).into()
    }
}

/// The maximum of two values, not requiring `Ord`. Provided for clarity in certain cases, though
/// `clamp_min` may sometimes be more clear.
pub fn max<T: PartialOrd + CubePrimitive>(lhs: T, rhs: T) -> T {
    clamp_min(lhs, rhs)
}

pub mod max {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Max).into()
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
                    binary_expand(scope, self.into(), rhs.into(), $operator).into()
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
                    let lhs: Variable = self.into();
                    let item = lhs.ty.with_vector_size(0);
                    binary_expand_fixed_output(scope, lhs, rhs.into(), item, $operator).into()
                }
            }
        }
    }
}

macro_rules! impl_binary_func_mixed_types {
    ($trait_name:ident, $method_name:ident, $rhs_ty: ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name<Rhs: CubePrimitive + CubeType<ExpandType: Into<Variable>> + Sized>:
                CubePrimitive + CubeType<ExpandType: [<$trait_name Expand>]<Rhs>> + Sized {
                fn $method_name(self, _rhs: Rhs) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &Scope,
                    lhs: NativeExpand<Self>,
                    rhs: NativeExpand<Rhs>,
                ) -> NativeExpand<Self> {
                    binary_expand(scope, lhs.into(), rhs.into(), $operator).into()
                }
            }

            pub trait [<$trait_name Expand>]<Rhs: CubeType>{
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: Rhs::ExpandType) -> Self;
            }

            $(impl $trait_name<$rhs_ty> for $type {})*
            impl<Rhs: CubePrimitive, T: CubePrimitive + $trait_name<Rhs>> [<$trait_name Expand>]<Rhs> for NativeExpand<T> {
                fn [<__expand_ $method_name _method>](self, scope: &Scope, rhs: NativeExpand<Rhs>) -> Self {
                    binary_expand(scope, self.into(), rhs.into(), $operator).into()
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
                    binary_expand(scope, self.into(), rhs.into(), $op).into()
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
                    assign_op_expand(scope, self, rhs, $op);
                }
            }
        }
    };
}

impl_core_binop!(Add, add, Arithmetic::Add);
impl_core_binop!(Sub, sub, Arithmetic::Sub);
impl_core_binop!(Mul, mul, Arithmetic::Mul);
impl_core_binop!(Div, div, Arithmetic::Div);
impl_core_binop!(Rem, rem, Arithmetic::Rem);

impl_core_assign_binop!(AddAssign, add_assign, Arithmetic::Add);
impl_core_assign_binop!(SubAssign, sub_assign, Arithmetic::Sub);
impl_core_assign_binop!(MulAssign, mul_assign, Arithmetic::Mul);
impl_core_assign_binop!(DivAssign, div_assign, Arithmetic::Div);
impl_core_assign_binop!(RemAssign, rem_assign, Arithmetic::Rem);

impl_core_binop!(BitAnd, bitand, Bitwise::BitwiseAnd);
impl_core_binop!(BitOr, bitor, Bitwise::BitwiseOr);
impl_core_binop!(BitXor, bitxor, Bitwise::BitwiseXor);
impl_core_binop!(Shl, shl, Bitwise::ShiftLeft);
impl_core_binop!(Shr, shr, Bitwise::ShiftRight);

impl_core_assign_binop!(BitAndAssign, bitand_assign, Bitwise::BitwiseAnd);
impl_core_assign_binop!(BitOrAssign, bitor_assign, Bitwise::BitwiseOr);
impl_core_assign_binop!(BitXorAssign, bitxor_assign, Bitwise::BitwiseXor);
impl_core_assign_binop!(ShlAssign, shl_assign, Bitwise::ShiftLeft);
impl_core_assign_binop!(ShrAssign, shr_assign, Bitwise::ShiftRight);

pub trait CubeAnd:
    CubePrimitive + Into<Variable> + CubeType<ExpandType: AndExpand> + Sized
{
    fn __expand_and_method(self, scope: &Scope, rhs: NativeExpand<Self>) -> NativeExpand<Self> {
        let this: Variable = self.into();
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
        binary_expand(scope, self.into(), rhs.into(), Operator::And).into()
    }
}

pub trait CubeOr: CubePrimitive + Into<Variable> + CubeType<ExpandType: AndExpand> + Sized {
    fn __expand_or_method(self, scope: &Scope, rhs: NativeExpand<Self>) -> NativeExpand<Self> {
        let this: Variable = self.into();
        let this: NativeExpand<Self> = this.into();
        this.__expand_and_method(scope, rhs)
    }
    fn __expand_or(
        scope: &Scope,
        lhs: NativeExpand<Self>,
        rhs: NativeExpand<Self>,
    ) -> NativeExpand<Self> {
        lhs.__expand_and_method(scope, rhs)
    }
}
pub trait OrExpand {
    fn __expand_or_method(self, scope: &Scope, rhs: Self) -> Self;
}

impl CubeOr for bool {}
impl<T: CubeOr + CubePrimitive> OrExpand for NativeExpand<T> {
    fn __expand_or_method(self, scope: &Scope, rhs: Self) -> Self {
        binary_expand(scope, self.into(), rhs.into(), Operator::Or).into()
    }
}

impl_binary_func!(
    Powf,
    powf,
    Arithmetic::Powf,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);

impl_binary_func!(
    Hypot,
    hypot,
    Arithmetic::Hypot,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);

impl_binary_func!(
    Rhypot,
    rhypot,
    Arithmetic::Rhypot,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);

impl_binary_func!(
    ArcTan2,
    atan2,
    Arithmetic::ArcTan2,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_binary_func!(
    ModFloor,
    mod_floor,
    Arithmetic::ModFloor,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64,
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
impl_binary_func!(MulHi, mul_hi, Arithmetic::MulHi, i32, u32, usize, isize);
impl_binary_func!(
    SaturatingAdd,
    saturating_add,
    Arithmetic::SaturatingAdd,
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
    Arithmetic::SaturatingSub,
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
    Dot,
    dot,
    Arithmetic::Dot,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64,
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

impl_binary_func_mixed_types!(
    Powi,
    powi,
    i32,
    Arithmetic::Powi,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64,
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
