use crate::frontend::operation::base::binary_expand;
use crate::frontend::{
    AtomicI32, AtomicI64, AtomicUInt, CubeContext, CubePrimitive, ExpandElementTyped, UInt, BF16,
    F16, F32, F64, I32, I64,
};
use crate::ir::Operator;
use crate::{frontend::CubeType, unexpanded};

macro_rules! impl_op {
    (($tr:ident|$func:ident|$op:tt) => { $($type:ty| $($rhs:ty);*),* }) => {
        $(
            $(
                impl $tr<$rhs> for $type {
                    type Output = Self;

                    fn $func(self, rhs: $rhs) -> Self::Output {
                        let rhs: Self = rhs.into();
                        self $op rhs
                    }
                }
            )*

            impl $tr for $type {
                type Output = Self;

                fn $func(self, rhs: Self) -> Self::Output {
                    (self.val $op rhs.val).into()
                }
            }
        )*
    };
}

pub mod add {
    use super::*;
    use core::ops::Add;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Add).into()
    }

    impl_op!(
        (Add|add|+) => {
            F16 | f32;u32,
            F32 | f32;u32,
            BF16 | f32;u32,
            F64 | f32;u32,
            I32 | i32;u32,
            I64 | i32;u32,
            UInt | u32
        }
    );
}

pub mod sub {
    use super::*;
    use core::ops::Sub;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Sub).into()
    }

    impl_op!(
        (Sub|sub|-) => {
            F16 | f32;u32,
            F32 | f32;u32,
            BF16 | f32;u32,
            F64 | f32;u32,
            I32 | i32;u32,
            I64 | i32;u32,
            UInt | u32
        }
    );
}

pub mod mul {
    use super::*;
    use core::ops::Mul;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Mul).into()
    }

    impl_op!(
        (Mul|mul|*) => {
            F16 | f32;u32,
            F32 | f32;u32,
            BF16 | f32;u32,
            F64 | f32;u32,
            I32 | i32;u32,
            I64 | i32;u32,
            UInt | u32
        }
    );
}

pub mod div {
    use super::*;
    use core::ops::Div;

    pub fn expand<C: CubePrimitive, R: Into<ExpandElementTyped<C>>>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: R,
    ) -> ExpandElementTyped<C> {
        let rhs: ExpandElementTyped<C> = rhs.into();
        binary_expand(context, lhs.into(), rhs.into(), Operator::Div).into()
    }

    impl_op!(
        (Div|div|/) => {
            F16 | f32;u32,
            F32 | f32;u32,
            BF16 | f32;u32,
            F64 | f32;u32,
            I32 | i32;u32,
            I64 | i32;u32,
            UInt | u32
        }
    );
}

pub mod rem {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Modulo).into()
    }

    macro_rules! impl_rem {
        ($type:ty) => {
            impl core::ops::Rem for $type {
                type Output = Self;

                fn rem(self, _rhs: Self) -> Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    impl_rem!(I32);
    impl_rem!(I64);
    impl_rem!(UInt);
}

pub mod and {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::And).into()
    }
}

pub mod bitand {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::BitwiseAnd).into()
    }

    impl core::ops::BitAnd for UInt {
        type Output = UInt;

        fn bitand(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

pub mod bitor {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::BitwiseOr).into()
    }

    impl core::ops::BitOr for UInt {
        type Output = UInt;

        fn bitor(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

pub mod or {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Or).into()
    }
}

pub mod bitxor {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::BitwiseXor).into()
    }

    impl core::ops::BitXor for UInt {
        type Output = UInt;

        fn bitxor(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

pub mod shl {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::ShiftLeft).into()
    }

    impl core::ops::Shl for UInt {
        type Output = UInt;

        fn shl(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

pub mod shr {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::ShiftRight).into()
    }

    impl core::ops::Shr for UInt {
        type Output = UInt;

        fn shr(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

/// For binary functions without special syntax
macro_rules! impl_binary_func {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $operator:expr, $($type:ty),*) => {
        pub trait $trait_name: CubeType + Sized {
            fn $method_name(self, _rhs: Self) -> Self {
                unexpanded!()
            }

            fn $method_name_expand(
                context: &mut CubeContext,
                lhs: ExpandElementTyped<Self>,
                rhs: ExpandElementTyped<Self>,
            ) -> ExpandElementTyped<Self> {
                binary_expand(context, lhs.into(), rhs.into(), $operator).into()
            }
        }

        $(impl $trait_name for $type {})*
    }
}

impl_binary_func!(
    Powf,
    powf,
    __expand_powf,
    Operator::Powf,
    F16,
    BF16,
    F32,
    F64
);
impl_binary_func!(
    Max,
    max,
    __expand_max,
    Operator::Max,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt,
    AtomicI32,
    AtomicI64,
    AtomicUInt
);
impl_binary_func!(
    Min,
    min,
    __expand_min,
    Operator::Min,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt
);
impl_binary_func!(
    Remainder,
    rem,
    __expand_rem,
    Operator::Remainder,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt
);
