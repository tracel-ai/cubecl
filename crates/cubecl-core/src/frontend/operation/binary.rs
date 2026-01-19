use crate::{
    flex32,
    frontend::{CubePrimitive, ExpandElementTyped},
    prelude::*,
};
use crate::{frontend::CubeType, tf32};
use crate::{
    frontend::operation::base::{binary_expand, binary_expand_fixed_output},
    unexpanded,
};
use crate::{
    ir::{Arithmetic, Bitwise, ExpandElement, Operator, Scope},
    prelude::assign_op_expand,
};
use core::ops::*;
use cubecl_common::{e2m1, e4m3, e5m2, ue8m0};
use cubecl_ir::ClampOperator;
use half::{bf16, f16};

pub mod add {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Add).into()
    }
}

pub mod sub {
    use cubecl_ir::{ConstantValue, Variable};

    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        // Dirty hack to enable slice destructuring with trailing patterns on `Sequence`
        match (lhs.expand.as_const(), rhs.expand.as_const()) {
            (Some(ConstantValue::UInt(lhs_val)), Some(ConstantValue::UInt(rhs_val))) => {
                let item_lhs = lhs.expand.ty;
                let item_rhs = rhs.expand.ty;

                let line_size = find_vectorization(item_lhs, item_rhs);

                let item = item_lhs.line(line_size);
                let value = (lhs_val - rhs_val).into();
                ExpandElement::Plain(Variable::constant(value, item)).into()
            }
            _ => binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Sub).into(),
        }
    }
}

pub mod mul {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Mul).into()
    }
}

pub mod div {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Div).into()
    }
}

pub mod rem {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Modulo).into()
    }
}

pub mod and {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        binary_expand(scope, lhs.into(), rhs.into(), Operator::And).into()
    }
}

pub mod bitand {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseAnd).into()
    }
}

pub mod bitor {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseOr).into()
    }
}

pub mod or {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        binary_expand(scope, lhs.into(), rhs.into(), Operator::Or).into()
    }
}

pub mod bitxor {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseXor).into()
    }
}

pub mod shl {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Bitwise::ShiftLeft).into()
    }
}

pub mod shr {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Bitwise::ShiftRight).into()
    }
}

pub mod clamp {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &mut Scope,
        input: ExpandElementTyped<C>,
        min: ExpandElementTyped<C>,
        max: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        unary_expand(scope, input.into(), |op| {
            Arithmetic::Clamp(ClampOperator {
                input: op.input,
                min_value: *min.expand,
                max_value: *max.expand,
            })
        })
        .into()
    }
}

pub mod clamp_max {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Min).into()
    }
}

pub mod clamp_min {
    use super::*;

    pub fn expand<C: PartialOrd + CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
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
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
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
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
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
                    scope: &mut Scope,
                    lhs: ExpandElementTyped<Self>,
                    rhs: ExpandElementTyped<Self>,
                ) -> ExpandElementTyped<Self> {
                    lhs.[<__expand_ $method_name _method>](scope, rhs)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: Self) -> Self;
            }

            $(impl $trait_name for $type {})*
            impl<T: CubePrimitive + $trait_name> [<$trait_name Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: Self) -> Self {
                    binary_expand(scope, self.into(), rhs.into(), $operator).into()
                }
            }
        }
    }
}

macro_rules! impl_binary_func_fixed_output_vectorization {
    ($trait_name:ident, $method_name:ident, $operator:expr, $out_vectorization: expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubePrimitive + CubeType<ExpandType: [<$trait_name Expand>]> + Sized {
                fn $method_name(self, _rhs: Self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &mut Scope,
                    lhs: ExpandElementTyped<Self>,
                    rhs: ExpandElementTyped<Self>,
                ) -> ExpandElementTyped<Self> {
                    lhs.[<__expand_ $method_name _method>](scope, rhs)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: Self) -> Self;
            }

            $(impl $trait_name for $type {})*
            impl<T: CubePrimitive + $trait_name> [<$trait_name Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: Self) -> Self {
                    let lhs: ExpandElement = self.into();
                    let item = lhs.ty.line($out_vectorization);
                    binary_expand_fixed_output(scope, lhs, rhs.into(), item, $operator).into()
                }
            }
        }
    }
}

macro_rules! impl_binary_func_mixed_types {
    ($trait_name:ident, $method_name:ident, $rhs_ty: ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name<Rhs: CubePrimitive + CubeType<ExpandType: Into<ExpandElement>> + Sized>:
                CubePrimitive + CubeType<ExpandType: [<$trait_name Expand>]<Rhs>> + Sized {
                fn $method_name(self, _rhs: Rhs) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &mut Scope,
                    lhs: ExpandElementTyped<Self>,
                    rhs: ExpandElementTyped<Rhs>,
                ) -> ExpandElementTyped<Self> {
                    binary_expand(scope, lhs.into(), rhs.into(), $operator).into()
                }
            }

            pub trait [<$trait_name Expand>]<Rhs: CubeType>{
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: Rhs::ExpandType) -> Self;
            }

            $(impl $trait_name<$rhs_ty> for $type {})*
            impl<Rhs: CubePrimitive, T: CubePrimitive + $trait_name<Rhs>> [<$trait_name Expand>]<Rhs> for ExpandElementTyped<T> {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: ExpandElementTyped<Rhs>) -> Self {
                    binary_expand(scope, self.into(), rhs.into(), $operator).into()
                }
            }
        }
    }
}

macro_rules! impl_core_binop {
    ($trait: ident, $method: ident, $op: expr) => {
        paste::paste! {
            pub trait [<Cube $trait>]: $trait<Output = Self> + CubePrimitive + CubeType<ExpandType: [<$trait Expand>]> + Sized {
                fn [<__expand_ $method>](
                    scope: &mut Scope,
                    lhs: ExpandElementTyped<Self>,
                    rhs: ExpandElementTyped<Self>,
                ) -> ExpandElementTyped<Self> {
                    lhs.[<__expand_ $method _method>](scope, rhs)
                }
            }

            pub trait [<$trait Expand>] {
                fn [<__expand_ $method _method>](self, scope: &mut Scope, rhs: Self) -> Self;
            }

            impl<T: $trait<Output = T> + CubePrimitive> [<Cube $trait>] for T {}
            impl<T: $trait<Output = T> + CubePrimitive> [<$trait Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method _method>](self, scope: &mut Scope, rhs: Self) -> Self {
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
                    scope: &mut Scope,
                    lhs: ExpandElementTyped<Self>,
                    rhs: ExpandElementTyped<Self>,
                ) {
                    lhs.[<__expand_ $method _method>](scope, rhs)
                }
            }

            pub trait [<$trait Expand>] {
                fn [<__expand_ $method _method>](self, scope: &mut Scope, rhs: Self);
            }

            impl<T: $trait + CubePrimitive> [<Cube $trait>] for T {}
            impl<T: $trait + CubePrimitive> [<$trait Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method _method>](self, scope: &mut Scope, rhs: Self) {
                    assign_op_expand(scope, self.into(), rhs.into(), $op);
                }
            }
        }
    };
}

impl_core_binop!(Add, add, Arithmetic::Add);
impl_core_binop!(Sub, sub, Arithmetic::Sub);
impl_core_binop!(Mul, mul, Arithmetic::Mul);
impl_core_binop!(Div, mul, Arithmetic::Div);
impl_core_binop!(Rem, rem, Arithmetic::Modulo);

impl_core_assign_binop!(AddAssign, add_assign, Arithmetic::Add);
impl_core_assign_binop!(SubAssign, sub_assign, Arithmetic::Sub);
impl_core_assign_binop!(MulAssign, mul_assign, Arithmetic::Mul);
impl_core_assign_binop!(DivAssign, div_assign, Arithmetic::Div);
impl_core_assign_binop!(RemAssign, rem_assign, Arithmetic::Modulo);

pub trait CubeOrd: Ord + CubeType<ExpandType: OrdExpand> + Sized {
    fn __expand_min(
        scope: &mut Scope,
        lhs: Self::ExpandType,
        rhs: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_min_method(scope, rhs)
    }

    fn __expand_max(
        scope: &mut Scope,
        lhs: Self::ExpandType,
        rhs: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_max_method(scope, rhs)
    }

    fn __expand_clamp(
        scope: &mut Scope,
        lhs: Self::ExpandType,
        min: Self::ExpandType,
        max: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_clamp_method(scope, min, max)
    }
}
pub trait OrdExpand {
    fn __expand_min_method(self, scope: &mut Scope, rhs: Self) -> Self;
    fn __expand_max_method(self, scope: &mut Scope, rhs: Self) -> Self;
    fn __expand_clamp_method(self, scope: &mut Scope, min: Self, max: Self) -> Self;
}

impl<T: Ord + CubePrimitive> CubeOrd for T {}
impl<T: Ord + CubePrimitive> OrdExpand for ExpandElementTyped<T> {
    fn __expand_min_method(self, scope: &mut Scope, rhs: Self) -> Self {
        binary_expand(scope, self.into(), rhs.into(), Arithmetic::Min).into()
    }
    fn __expand_max_method(self, scope: &mut Scope, rhs: Self) -> Self {
        binary_expand(scope, self.into(), rhs.into(), Arithmetic::Max).into()
    }
    fn __expand_clamp_method(self, scope: &mut Scope, min: Self, max: Self) -> Self {
        unary_expand(scope, self.into(), |op| {
            Arithmetic::Clamp(ClampOperator {
                input: op.input,
                min_value: *min.expand,
                max_value: *max.expand,
            })
        })
        .into()
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
    Remainder,
    rem,
    Arithmetic::Remainder,
    e2m1,
    e4m3,
    e5m2,
    ue8m0,
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
impl_binary_func_fixed_output_vectorization!(
    Dot,
    dot,
    Arithmetic::Dot,
    0,
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
