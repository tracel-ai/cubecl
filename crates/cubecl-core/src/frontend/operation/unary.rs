use cubecl_ir::{Bitwise, Operator};
use half::{bf16, f16};

use crate::{
    flex32,
    ir::{Arithmetic, ExpandElement, Scope},
    prelude::{CubePrimitive, ExpandElementTyped},
    tf32, unexpanded,
};

use super::base::{unary_expand, unary_expand_fixed_output};

pub mod not {
    use super::*;

    pub fn expand(scope: &mut Scope, x: ExpandElementTyped<bool>) -> ExpandElementTyped<bool> {
        unary_expand(scope, x.into(), Operator::Not).into()
    }
}

pub mod neg {
    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        x: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        unary_expand(scope, x.into(), Arithmetic::Neg).into()
    }
}

macro_rules! impl_unary_func {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $operator:expr, $($type:ty),*) => {
        pub trait $trait_name: CubePrimitive + Sized {
            #[allow(unused_variables)]
            fn $method_name(x: Self) -> Self {
                unexpanded!()
            }

            fn $method_name_expand(scope: &mut Scope, x: Self::ExpandType) -> ExpandElementTyped<Self> {
                unary_expand(scope, x.into(), $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
    }
}

impl Exp for f32 {
    fn exp(x: Self) -> Self {
        x.exp()
    }
}

macro_rules! impl_unary_func_fixed_out_vectorization {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $operator:expr, $out_vectorization: expr, $($type:ty),*) => {
        pub trait $trait_name: CubePrimitive + Sized {
            #[allow(unused_variables)]
            fn $method_name(x: Self) -> Self {
                unexpanded!()
            }

            fn $method_name_expand(scope: &mut Scope, x: Self::ExpandType) -> ExpandElementTyped<Self> {
                let expand_element: ExpandElement = x.into();
                let mut item = expand_element.item;
                item.vectorization = $out_vectorization;
                unary_expand_fixed_output(scope, expand_element, item, $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
    }
}

macro_rules! impl_unary_func_fixed_out_ty {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $out_ty: ty, $operator:expr, $($type:ty),*) => {
        pub trait $trait_name: CubePrimitive + Sized {
            #[allow(unused_variables)]
            fn $method_name(x: Self) -> $out_ty {
                unexpanded!()
            }

            fn $method_name_expand(scope: &mut Scope, x: Self::ExpandType) -> ExpandElementTyped<$out_ty> {
                let expand_element: ExpandElement = x.into();
                let mut item = expand_element.item;
                item.elem = <$out_ty as CubePrimitive>::as_elem(scope);
                unary_expand_fixed_output(scope, expand_element, item, $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
    }
}

impl_unary_func!(
    Abs,
    abs,
    __expand_abs,
    Arithmetic::Abs,
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
    u64
);
impl_unary_func!(
    Exp,
    exp,
    __expand_exp,
    Arithmetic::Exp,
    f16,
    bf16,
    flex32,
    tf32,
    // f32,
    f64
);
impl_unary_func!(
    Log,
    log,
    __expand_log,
    Arithmetic::Log,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Log1p,
    log1p,
    __expand_log1p,
    Arithmetic::Log1p,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Cos,
    cos,
    __expand_cos,
    Arithmetic::Cos,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Sin,
    sin,
    __expand_sin,
    Arithmetic::Sin,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Tanh,
    tanh,
    __expand_tanh,
    Arithmetic::Tanh,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Sqrt,
    sqrt,
    __expand_sqrt,
    Arithmetic::Sqrt,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Round,
    round,
    __expand_round,
    Arithmetic::Round,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Floor,
    floor,
    __expand_floor,
    Arithmetic::Floor,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Ceil,
    ceil,
    __expand_ceil,
    Arithmetic::Ceil,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Erf,
    erf,
    __expand_erf,
    Arithmetic::Erf,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Recip,
    recip,
    __expand_recip,
    Arithmetic::Recip,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func_fixed_out_vectorization!(
    Magnitude,
    magnitude,
    __expand_magnitude,
    Arithmetic::Magnitude,
    None,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Normalize,
    normalize,
    __expand_normalize,
    Arithmetic::Normalize,
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
    __expand_count_ones,
    u32,
    Bitwise::CountOnes,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64
);
impl_unary_func!(
    ReverseBits,
    reverse_bits,
    __expand_reverse_bits,
    Bitwise::ReverseBits,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64
);

impl_unary_func!(
    BitwiseNot,
    bitwise_not,
    __expand_bitwise_not,
    Bitwise::BitwiseNot,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64
);
impl_unary_func_fixed_out_ty!(
    LeadingZeros,
    leading_zeros,
    __expand_leading_zeros,
    u32,
    Bitwise::LeadingZeros,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64
);
impl_unary_func_fixed_out_ty!(
    FindFirstSet,
    find_first_set,
    __expand_find_first_set,
    u32,
    Bitwise::FindFirstSet,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64
);
