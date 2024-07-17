use crate::frontend::operation::base::cmp_expand;
use crate::frontend::{CubeContext, ExpandElementTyped, UInt, BF16, F16, F32, F64, I32, I64};
use crate::ir::Operator;
use crate::prelude::CubePrimitive;

macro_rules! impl_cmp {
    ({ $($type:ty| $($rhs:ty);*),* }) => {
        $(
            $(
                impl core::cmp::PartialEq<$rhs> for $type {
                    fn eq(&self, rhs: &$rhs) -> bool {
                        let rhs: Self = (*rhs).into();
                        self == &rhs
                    }
                }

                impl core::cmp::PartialOrd<$rhs> for $type {
                    fn partial_cmp(&self, rhs: &$rhs) -> Option<std::cmp::Ordering> {
                        let rhs: Self = (*rhs).into();
                        core::cmp::PartialOrd::partial_cmp(self, &rhs)
                    }
                }

            )*

            impl_cmp!($type);
        )*
    };
    ($type:ty) => {
        impl core::cmp::PartialEq for $type {
            fn eq(&self, other: &Self) -> bool {
                self.val == other.val && self.vectorization == other.vectorization
            }
        }

        impl core::cmp::Eq for $type {}

        impl core::cmp::PartialOrd for $type {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                match self.val.partial_cmp(&other.val) {
                    Some(core::cmp::Ordering::Equal) => {}
                    ord => return ord,
                }
                self.vectorization.partial_cmp(&other.vectorization)
            }
        }
    };
}

impl_cmp!(
    {
        F16 | f32;u32,
        F32 | f32;u32,
        BF16 | f32;u32,
        F64 | f32;u32,
        I32 | i32;u32,
        I64 | i32;u32,
        UInt | u32

    }
);

pub mod ne {

    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Operator::NotEqual).into()
    }
}

pub mod gt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Operator::Greater).into()
    }
}

pub mod lt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Operator::Lower).into()
    }
}

pub mod ge {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Operator::GreaterEqual).into()
    }
}

pub mod le {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Operator::LowerEqual).into()
    }
}

pub mod eq {

    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Operator::Equal).into()
    }
}

pub mod add_assign {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        cmp_expand(context, lhs.into(), rhs.into(), Operator::Add).into()
    }
}
