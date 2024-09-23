use crate::frontend::CubeType;
use crate::frontend::{CubeContext, CubePrimitive, ExpandElement, ExpandElementTyped};
use crate::ir::Operator;
use crate::{
    frontend::operation::base::{binary_expand, binary_expand_fixed_output},
    unexpanded,
};
use half::{bf16, f16};

pub mod add {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into().into(), rhs.into().into(), Operator::Add).into()
    }
}

pub mod sub {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into().into(), rhs.into().into(), Operator::Sub).into()
    }
}

pub mod mul {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into().into(), rhs.into().into(), Operator::Mul).into()
    }
}

pub mod div {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into().into(), rhs.into().into(), Operator::Div).into()
    }
}

pub mod rem {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::Modulo,
        )
        .into()
    }
}

pub mod and {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        binary_expand(context, lhs.into().into(), rhs.into().into(), Operator::And).into()
    }
}

pub mod bitand {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::BitwiseAnd,
        )
        .into()
    }
}

pub mod bitor {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::BitwiseOr,
        )
        .into()
    }
}

pub mod or {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        binary_expand(context, lhs.into().into(), rhs.into().into(), Operator::Or).into()
    }
}

pub mod bitxor {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::BitwiseXor,
        )
        .into()
    }
}

pub mod shl {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::ShiftLeft,
        )
        .into()
    }
}

pub mod shr {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        binary_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::ShiftRight,
        )
        .into()
    }
}

/// For binary functions without special syntax
macro_rules! impl_binary_func {
    ($trait_name:ident, $method_name:ident, $func_name_expand:ident, $method_name_expand:ident, $operator:expr, $($type:ty),*) => {
        pub trait $trait_name: CubeType + Sized {
            fn $method_name(self, _rhs: Self) -> Self {
                unexpanded!()
            }

            fn $func_name_expand(
                context: &mut CubeContext,
                lhs: ExpandElementTyped<Self>,
                rhs: ExpandElementTyped<Self>,
            ) -> ExpandElementTyped<Self> {
                binary_expand(context, lhs.into(), rhs.into(), $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
        $(impl ExpandElementTyped<$type> {
            pub fn $method_name_expand(self, context: &mut CubeContext, rhs: ExpandElementTyped<$type>) -> ExpandElementTyped<$type> {
                binary_expand(context, self.into(), rhs.into(), $operator).into()
            }
        })*
    }
}

macro_rules! impl_binary_func_fixed_output_vectorization {
    ($trait_name:ident, $method_name:ident, $func_name_expand:ident, $method_name_expand:ident, $operator:expr, $out_vectorization: expr, $($type:ty),*) => {
        pub trait $trait_name: CubeType + Sized {
            fn $method_name(self, _rhs: Self) -> Self {
                unexpanded!()
            }

            fn $func_name_expand(
                context: &mut CubeContext,
                lhs: ExpandElementTyped<Self>,
                rhs: ExpandElementTyped<Self>,
            ) -> ExpandElementTyped<Self> {
                let lhs: ExpandElement = lhs.into();
                let mut item = lhs.item();
                item.vectorization = $out_vectorization;
                binary_expand_fixed_output(context, lhs, rhs.into(), item, $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
        $(impl ExpandElementTyped<$type> {
            pub fn $method_name_expand(self, context: &mut CubeContext, rhs: ExpandElementTyped<$type>) -> ExpandElementTyped<$type> {
                let lhs: ExpandElement = self.into();
                let mut item = lhs.item();
                item.vectorization = $out_vectorization;
                binary_expand_fixed_output(context, lhs, rhs.into(), item, $operator).into()
            }
        })*
    }
}

impl_binary_func!(
    Powf,
    powf,
    __expand_powf,
    __expand_powf_method,
    Operator::Powf,
    f16,
    bf16,
    f32,
    f64
);
impl_binary_func!(
    Max,
    max,
    __expand_max,
    __expand_max_method,
    Operator::Max,
    f16,
    bf16,
    f32,
    f64,
    i32,
    i64,
    u32
);
impl_binary_func!(
    Min,
    min,
    __expand_min,
    __expand_min_method,
    Operator::Min,
    f16,
    bf16,
    f32,
    f64,
    i32,
    i64,
    u32
);
impl_binary_func!(
    Remainder,
    rem,
    __expand_rem,
    __expand_rem_method,
    Operator::Remainder,
    f16,
    bf16,
    f32,
    f64,
    i32,
    i64,
    u32
);
impl_binary_func_fixed_output_vectorization!(
    Dot,
    dot,
    __expand_dot,
    __expand_dot_method,
    Operator::Dot,
    None,
    f16,
    bf16,
    f32,
    f64,
    i32,
    i64,
    u32
);
