use core::ops::Not;
use cubecl_common::{e2m1, e4m3, e5m2, ue8m0};
use cubecl_ir::{Bitwise, Comparison, Operator, Type};
use half::{bf16, f16};

use crate::{
    flex32,
    ir::{Arithmetic, ExpandElement, Scope},
    prelude::{CubePrimitive, CubeType, ExpandElementTyped},
    tf32, unexpanded,
};

use super::base::{unary_expand, unary_expand_fixed_output};

pub mod not {
    use super::*;

    pub fn expand<T: CubeNot>(
        scope: &mut Scope,
        x: ExpandElementTyped<T>,
    ) -> ExpandElementTyped<T> {
        if x.expand.ty.is_bool() {
            unary_expand(scope, x.into(), Operator::Not).into()
        } else {
            unary_expand(scope, x.into(), Bitwise::BitwiseNot).into()
        }
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
    ($trait_name:ident, $method_name:ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubeType<ExpandType: [<$trait_name Expand>]> + Sized {
                #[allow(unused_variables)]
                fn $method_name(self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](scope: &mut Scope, x: Self::ExpandType) -> Self::ExpandType {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> Self;
            }

            $(impl $trait_name for $type {})*
            impl<T: $trait_name + CubePrimitive> [<$trait_name Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> Self {
                    unary_expand(scope, self.into(), $operator).into()
                }
            }
        }
    }
}

impl Exp for f32 {
    fn exp(self) -> Self {
        self.exp()
    }
}

macro_rules! impl_unary_func_fixed_out_vectorization {
    ($trait_name:ident, $method_name:ident, $operator:expr, $out_vectorization: expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubeType<ExpandType: [<$trait_name Expand>]> + Sized {
                #[allow(unused_variables)]
                fn $method_name(self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](scope: &mut Scope, x: Self::ExpandType) -> Self::ExpandType {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> Self;
            }

            $(impl $trait_name for $type {})*
            impl<T: $trait_name + CubePrimitive> [<$trait_name Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> Self {
                    let expand_element: ExpandElement = self.into();
                    let item = expand_element.ty.line($out_vectorization);
                    unary_expand_fixed_output(scope, expand_element, item, $operator).into()
                }
            }
        }
    }
}

macro_rules! impl_unary_func_fixed_out_ty {
    ($trait_name:ident, $method_name:ident, $out_ty: ty, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubeType<ExpandType: [<$trait_name Expand>]> + Sized {
                #[allow(unused_variables, clippy::wrong_self_convention)]
                fn $method_name(self) -> $out_ty {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](scope: &mut Scope, x: Self::ExpandType) -> ExpandElementTyped<$out_ty> {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> ExpandElementTyped<$out_ty>;
            }

            $(impl $trait_name for $type {})*
            impl<T: $trait_name + CubePrimitive> [<$trait_name Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> ExpandElementTyped<$out_ty> {
                    let expand_element: ExpandElement = self.into();
                    let item = Type::new(<$out_ty as CubePrimitive>::as_type(scope)).line(expand_element.ty.line_size());
                    unary_expand_fixed_output(scope, expand_element, item, $operator).into()
                }
            }
        }
    }
}

// Needs special handling because Rust combines bitwise and logical or into one trait
macro_rules! impl_not {
    ($trait_name:ident, $method_name:ident, $($type:ty),*) => {
        paste::paste! {
            pub trait [<Cube $trait_name>]: $trait_name<Output = Self> + CubeType<ExpandType: [<$trait_name Expand>]> {
                fn [<__expand_ $method_name>](scope: &mut Scope, x: Self::ExpandType) -> Self::ExpandType {
                    x.[<__expand_ $method_name _method>](scope)
                }
            }

            pub trait [<$trait_name Expand>] {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> Self;
            }

            $(impl [<Cube $trait_name>] for $type {})*
            impl<T: [<Cube $trait_name>] + CubePrimitive> [<$trait_name Expand>] for ExpandElementTyped<T> {
                fn [<__expand_ $method_name _method>](self, scope: &mut Scope) -> Self {
                    not::expand(scope, self.into())
                }
            }
        }
    }
}

impl_not!(Not, not, bool, u8, u16, u32, u64, i8, i16, i32, i64);

impl_unary_func!(
    Abs,
    abs,
    Arithmetic::Abs,
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
    u64
);
impl_unary_func!(
    Exp,
    exp,
    Arithmetic::Exp,
    f16,
    bf16,
    flex32,
    tf32,
    // f32,
    f64
);
impl_unary_func!(Log, ln, Arithmetic::Log, f16, bf16, flex32, tf32, f32, f64);
impl_unary_func!(
    Log1p,
    log1p,
    Arithmetic::Log1p,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(Cos, cos, Arithmetic::Cos, f16, bf16, flex32, tf32, f32, f64);
impl_unary_func!(Sin, sin, Arithmetic::Sin, f16, bf16, flex32, tf32, f32, f64);
impl_unary_func!(Tan, tan, Arithmetic::Tan, f16, bf16, flex32, tf32, f32, f64);
impl_unary_func!(
    Tanh,
    tanh,
    Arithmetic::Tanh,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Sinh,
    sinh,
    Arithmetic::Sinh,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Cosh,
    cosh,
    Arithmetic::Cosh,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    ArcCos,
    acos,
    Arithmetic::ArcCos,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    ArcSin,
    asin,
    Arithmetic::ArcSin,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    ArcTan,
    atan,
    Arithmetic::ArcTan,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    ArcSinh,
    asinh,
    Arithmetic::ArcSinh,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    ArcCosh,
    acosh,
    Arithmetic::ArcCosh,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    ArcTanh,
    atanh,
    Arithmetic::ArcTanh,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Degrees,
    to_degrees,
    Arithmetic::Degrees,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Radians,
    to_radians,
    Arithmetic::Radians,
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
    Arithmetic::Sqrt,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    InverseSqrt,
    inverse_sqrt,
    Arithmetic::InverseSqrt,
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
    Arithmetic::Ceil,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(
    Trunc,
    trunc,
    Arithmetic::Trunc,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func!(Erf, erf, Arithmetic::Erf, f16, bf16, flex32, tf32, f32, f64);
impl_unary_func!(
    Recip,
    recip,
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
    Arithmetic::Magnitude,
    0,
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

impl_unary_func_fixed_out_ty!(
    LeadingZeros,
    leading_zeros,
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
impl_unary_func_fixed_out_ty!(
    IsNan,
    is_nan,
    bool,
    Comparison::IsNan,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
impl_unary_func_fixed_out_ty!(
    IsInf,
    is_inf,
    bool,
    Comparison::IsInf,
    f16,
    bf16,
    flex32,
    tf32,
    f32,
    f64
);
