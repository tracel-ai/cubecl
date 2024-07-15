use crate::frontend::operation::base::binary_expand;
use crate::frontend::{
    CubeContext, CubePrimitive, ExpandElementTyped, UInt, BF16, F16, F32, F64, I32,
    I64,
};
use crate::ir::Operator;
use crate::{frontend::CubeType, unexpanded};

pub mod add {

    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Add).into()
    }

    macro_rules! impl_add {
        ($type:ty) => {
            impl core::ops::Add for $type {
                type Output = Self;

                fn add(self, rhs: Self) -> Self::Output {
                    (self.val + rhs.val).into()
                }
            }
        };
    }

    impl_add!(F16);
    impl_add!(BF16);
    impl_add!(F32);
    impl_add!(F64);
    impl_add!(I32);
    impl_add!(I64);
    impl_add!(UInt);
}

pub mod sub {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Sub).into()
    }

    macro_rules! impl_sub {
        ($type:ty) => {
            impl core::ops::Sub for $type {
                type Output = Self;

                fn sub(self, rhs: Self) -> Self::Output {
                    (self.val - rhs.val).into()
                }
            }
        };
    }

    impl_sub!(F16);
    impl_sub!(BF16);
    impl_sub!(F32);
    impl_sub!(F64);
    impl_sub!(I32);
    impl_sub!(I64);
    impl_sub!(UInt);
}

pub mod mul {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Mul).into()
    }

    macro_rules! impl_mul {
        ($type:ty) => {
            impl core::ops::Mul for $type {
                type Output = Self;

                fn mul(self, rhs: Self) -> Self::Output {
                    (self.val * rhs.val).into()
                }
            }
        };
    }

    impl_mul!(F16);
    impl_mul!(BF16);
    impl_mul!(F32);
    impl_mul!(F64);
    impl_mul!(I32);
    impl_mul!(I64);
    impl_mul!(UInt);
}

pub mod div {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Div).into()
    }

    macro_rules! impl_div {
        ($type:ty) => {
            impl core::ops::Div for $type {
                type Output = Self;

                fn div(self, rhs: Self) -> Self::Output {
                    (self.val / rhs.val).into()
                }
            }
        };
    }

    impl_div!(F16);
    impl_div!(BF16);
    impl_div!(F32);
    impl_div!(F64);
    impl_div!(I32);
    impl_div!(I64);
    impl_div!(UInt);
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
    UInt
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
