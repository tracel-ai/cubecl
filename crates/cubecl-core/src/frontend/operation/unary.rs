use half::{bf16, f16};

use crate::{
    frontend::CubeContext,
    ir::Operator,
    prelude::{CubePrimitive, ExpandElement, ExpandElementTyped},
    unexpanded,
};

use super::base::{unary_expand, unary_expand_fixed_output};

pub mod not {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        x: ExpandElementTyped<bool>,
    ) -> ExpandElementTyped<bool> {
        unary_expand(context, x.into(), Operator::Not).into()
    }
}

pub mod neg {
    use crate::prelude::Numeric;

    use super::*;

    pub fn expand<N: Numeric>(
        context: &mut CubeContext,
        x: ExpandElementTyped<N>,
    ) -> ExpandElementTyped<N> {
        unary_expand(context, x.into(), Operator::Neg).into()
    }
}

macro_rules! impl_unary_func {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $operator:expr, $($type:ty),*) => {
        pub trait $trait_name: CubePrimitive + Sized {
            #[allow(unused_variables)]
            fn $method_name(x: Self) -> Self {
                unexpanded!()
            }

            fn $method_name_expand(context: &mut CubeContext, x: Self::ExpandType) -> ExpandElementTyped<Self> {
                unary_expand(context, x.into(), $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
    }
}

macro_rules! impl_unary_func_fixed_out_vectorization {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $operator:expr, $out_vectorization: expr, $($type:ty),*) => {
        pub trait $trait_name: CubePrimitive + Sized {
            #[allow(unused_variables)]
            fn $method_name(x: Self) -> Self {
                unexpanded!()
            }

            fn $method_name_expand(context: &mut CubeContext, x: Self::ExpandType) -> ExpandElementTyped<Self> {
                let expand_element: ExpandElement = x.into();
                let mut item = expand_element.item();
                item.vectorization = $out_vectorization;
                unary_expand_fixed_output(context, expand_element, item, $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
    }
}

impl_unary_func!(
    Abs,
    abs,
    __expand_abs,
    Operator::Abs,
    f16,
    bf16,
    f32,
    f64,
    i32,
    i64,
    u32
);
impl_unary_func!(Exp, exp, __expand_exp, Operator::Exp, f16, bf16, f32, f64);
impl_unary_func!(Log, log, __expand_log, Operator::Log, f16, bf16, f32, f64);
impl_unary_func!(
    Log1p,
    log1p,
    __expand_log1p,
    Operator::Log1p,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func!(Cos, cos, __expand_cos, Operator::Cos, f16, bf16, f32, f64);
impl_unary_func!(Sin, sin, __expand_sin, Operator::Sin, f16, bf16, f32, f64);
impl_unary_func!(
    Tanh,
    tanh,
    __expand_tanh,
    Operator::Tanh,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func!(
    Sqrt,
    sqrt,
    __expand_sqrt,
    Operator::Sqrt,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func!(
    Round,
    round,
    __expand_round,
    Operator::Round,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func!(
    Floor,
    floor,
    __expand_floor,
    Operator::Floor,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func!(
    Ceil,
    ceil,
    __expand_ceil,
    Operator::Ceil,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func!(Erf, erf, __expand_erf, Operator::Erf, f16, bf16, f32, f64);
impl_unary_func!(
    Recip,
    recip,
    __expand_recip,
    Operator::Recip,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func_fixed_out_vectorization!(
    Magnitude,
    magnitude,
    __expand_magnitude,
    Operator::Magnitude,
    None,
    f16,
    bf16,
    f32,
    f64
);
impl_unary_func!(
    Normalize,
    normalize,
    __expand_normalize,
    Operator::Normalize,
    f16,
    bf16,
    f32,
    f64
);
