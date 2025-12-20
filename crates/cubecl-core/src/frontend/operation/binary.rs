use crate::ir::{Arithmetic, Bitwise, ExpandElement, Operator, Scope};
use crate::{
    flex32,
    frontend::{CubePrimitive, ExpandElementTyped},
};
use crate::{frontend::CubeType, tf32};
use crate::{
    frontend::operation::base::{binary_expand, binary_expand_fixed_output},
    unexpanded,
};
use cubecl_common::{e2m1, e4m3, e5m2, ue8m0};
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
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(scope, lhs.into(), rhs.into(), Arithmetic::Sub).into()
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

/// For binary functions without special syntax
macro_rules! impl_binary_func {
    ($trait_name:ident, $method_name:ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubeType + Sized {
                fn $method_name(self, _rhs: Self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &mut Scope,
                    lhs: ExpandElementTyped<Self>,
                    rhs: ExpandElementTyped<Self>,
                ) -> ExpandElementTyped<Self> {
                    binary_expand(scope, lhs.into(), rhs.into(), $operator).into()
                }
            }

            $(impl $trait_name for $type {})*
            $(impl ExpandElementTyped<$type> {
                pub fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: ExpandElementTyped<$type>) -> ExpandElementTyped<$type> {
                    binary_expand(scope, self.into(), rhs.into(), $operator).into()
                }
            })*
        }
    }
}

macro_rules! impl_binary_func_fixed_output_vectorization {
    ($trait_name:ident, $method_name:ident, $operator:expr, $out_vectorization: expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name: CubeType + Sized {
                fn $method_name(self, _rhs: Self) -> Self {
                    unexpanded!()
                }

                fn [<__expand_ $method_name>](
                    scope: &mut Scope,
                    lhs: ExpandElementTyped<Self>,
                    rhs: ExpandElementTyped<Self>,
                ) -> ExpandElementTyped<Self> {
                    let lhs: ExpandElement = lhs.into();
                    let item = lhs.ty.line($out_vectorization);
                    binary_expand_fixed_output(scope, lhs, rhs.into(), item, $operator).into()
                }
            }

            $(impl $trait_name for $type {})*
            $(impl ExpandElementTyped<$type> {
                pub fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: ExpandElementTyped<$type>) -> ExpandElementTyped<$type> {
                    let lhs: ExpandElement = self.into();
                    let item = lhs.ty.line($out_vectorization);
                    binary_expand_fixed_output(scope, lhs, rhs.into(), item, $operator).into()
                }
            })*
        }
    }
}

macro_rules! impl_binary_func_mixed_types {
    ($trait_name:ident, $method_name:ident, $rhs_ty: ident, $operator:expr, $($type:ty),*) => {
        paste::paste! {
            pub trait $trait_name<Rhs: CubeType + Sized>: CubeType + Sized {
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

            $(impl $trait_name<$rhs_ty> for $type {})*
            $(impl ExpandElementTyped<$type> {
                pub fn [<__expand_ $method_name _method>](self, scope: &mut Scope, rhs: ExpandElementTyped<$rhs_ty>) -> ExpandElementTyped<$type> {
                    binary_expand(scope, self.into(), rhs.into(), $operator).into()
                }
            })*
        }
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
    Max,
    max,
    Arithmetic::Max,
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
    usize
);
impl_binary_func!(
    Min,
    min,
    Arithmetic::Min,
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
    usize
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
    usize
);
impl_binary_func!(MulHi, mul_hi, Arithmetic::MulHi, i32, u32, usize);
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
    usize
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
    usize
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
    usize
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
    usize
);
