use crate::frontend::{Array, CubeContext, ExpandElement, SharedMemory, Tensor, UInt};
use crate::frontend::{BF16, F16, F32, F64, I32, I64};
use crate::{ir, unexpanded};

macro_rules! impl_op_assign {
    (($tr:ident|$func:ident) => { $($type:ty| $($rhs:ty);*),* }) => {
        $(
            $(
                impl $tr<$rhs> for $type {
                    fn $func(&mut self, _rhs: $rhs) {
                        unexpanded!()
                    }
                }
            )*

            impl $tr for $type {
                fn $func(&mut self, _rhs: Self) {
                    unexpanded!()
                }
            }
        )*
    };
}

pub mod assign {
    use self::ir::{Operator, UnaryOperator};

    use super::*;

    pub fn expand<I: Into<ExpandElement>, O: Into<ExpandElement>>(
        context: &mut CubeContext,
        input: I,
        output: O,
    ) {
        context.register(Operator::Assign(UnaryOperator {
            input: *input.into(),
            out: *output.into(),
        }));
    }
}

pub mod index_assign {
    use crate::{
        frontend::CubeType,
        prelude::{ExpandElementTyped, SliceMut},
        unexpanded,
    };

    use self::ir::{BinaryOperator, Operator, Variable};

    use super::*;

    pub fn expand<A: CubeType + core::ops::Index<UInt>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<UInt>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        let index: Variable = index.expand.into();
        let index = match index {
            Variable::ConstantScalar(value) => {
                Variable::ConstantScalar(ir::ConstantScalarValue::UInt(value.as_u64()))
            }
            _ => index,
        };
        context.register(Operator::IndexAssign(BinaryOperator {
            lhs: index,
            rhs: value.expand.into(),
            out: array.expand.into(),
        }));
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubeType, I: Into<UInt>> core::ops::IndexMut<I> for $type<E> {
                fn index_mut(&mut self, _index: I) -> &mut Self::Output {
                    unexpanded!()
                }
            }
        };
    }
    macro_rules! impl_index_vec {
        ($($type:ident),*) => {
            $(
                impl core::ops::IndexMut<UInt> for $type {
                    fn index_mut(&mut self, _index: UInt) -> &mut Self::Output {
                        unexpanded!()
                    }
                }
                impl core::ops::IndexMut<u32> for $type {
                    fn index_mut(&mut self, _index: u32) -> &mut Self::Output {
                        unexpanded!()
                    }
                }

            )*
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
    impl_index_vec!(I64, I32, F16, BF16, F32, F64, UInt);

    impl<'a, E: CubeType, I: Into<UInt>> core::ops::IndexMut<I> for SliceMut<'a, E> {
        fn index_mut(&mut self, _index: I) -> &mut Self::Output {
            unexpanded!()
        }
    }
}

pub mod index {
    use crate::{
        frontend::{
            operation::base::{binary_expand, binary_expand_no_vec},
            CubeType,
        },
        prelude::{ExpandElementTyped, Slice, SliceMut},
        unexpanded,
    };

    use self::ir::{Operator, Variable};

    use super::*;

    pub fn expand<A: CubeType + core::ops::Index<UInt>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<UInt>,
    ) -> ExpandElementTyped<A::Output>
    where
        A::Output: CubeType + Sized,
    {
        let index: ExpandElement = index.into();
        let index_var: Variable = *index;
        let index = match index_var {
            Variable::ConstantScalar(value) => ExpandElement::Plain(Variable::ConstantScalar(
                ir::ConstantScalarValue::UInt(value.as_u64()),
            )),
            _ => index,
        };
        let array: ExpandElement = array.into();
        let var: Variable = *array;
        let var = match var {
            Variable::Local { .. } => binary_expand_no_vec(context, array, index, Operator::Index),
            _ => binary_expand(context, array, index, Operator::Index),
        };

        ExpandElementTyped::new(var)
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubeType, I: Into<UInt>> core::ops::Index<I> for $type<E> {
                type Output = E;

                fn index(&self, _index: I) -> &Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    macro_rules! impl_index_vec {
        ($($type:ident),*) => {
            $(
                impl core::ops::Index<UInt> for $type {
                    type Output = Self;

                    fn index(&self, _index: UInt) -> &Self::Output {
                        unexpanded!()
                    }
                }

                impl core::ops::Index<u32> for $type {
                    type Output = Self;

                    fn index(&self, _index: u32) -> &Self::Output {
                        unexpanded!()
                    }
                }
            )*
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);

    impl_index_vec!(I64, I32, F16, BF16, F32, F64, UInt);

    impl<'a, E: CubeType, I: Into<UInt>> core::ops::Index<I> for SliceMut<'a, E> {
        type Output = E;
        fn index(&self, _index: I) -> &Self::Output {
            unexpanded!()
        }
    }

    impl<'a, E: CubeType, I: Into<UInt>> core::ops::Index<I> for Slice<'a, E> {
        type Output = E;
        fn index(&self, _index: I) -> &Self::Output {
            unexpanded!()
        }
    }
}

pub mod add_assign_array_op {
    use self::ir::Operator;
    use super::*;
    use crate::prelude::{array_assign_binary_op_expand, CubeType, ExpandElementTyped};

    pub fn expand<A: CubeType + core::ops::Index<UInt>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<UInt>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(context, array, index, value, Operator::Add);
    }
}

pub mod sub_assign_array_op {
    use self::ir::Operator;
    use super::*;
    use crate::prelude::{array_assign_binary_op_expand, CubeType, ExpandElementTyped};

    pub fn expand<A: CubeType + core::ops::Index<UInt>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<UInt>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(context, array, index, value, Operator::Sub);
    }
}

pub mod mul_assign_array_op {
    use self::ir::Operator;
    use super::*;
    use crate::prelude::{array_assign_binary_op_expand, CubeType, ExpandElementTyped};

    pub fn expand<A: CubeType + core::ops::Index<UInt>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<UInt>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(context, array, index, value, Operator::Mul);
    }
}

pub mod div_assign_array_op {
    use self::ir::Operator;
    use super::*;
    use crate::prelude::{array_assign_binary_op_expand, CubeType, ExpandElementTyped};

    pub fn expand<A: CubeType + core::ops::Index<UInt>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<UInt>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(context, array, index, value, Operator::Div);
    }
}

pub mod add_assign_op {
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};
    use core::ops::AddAssign;

    use self::ir::Operator;

    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Add)
    }

    impl_op_assign!(
        (AddAssign|add_assign) => {
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

pub mod sub_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};
    use core::ops::SubAssign;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Sub)
    }

    impl_op_assign!(
        (SubAssign|sub_assign) => {
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

pub mod mul_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};
    use core::ops::MulAssign;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Mul)
    }

    impl_op_assign!(
        (MulAssign|mul_assign) => {
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

pub mod div_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};
    use core::ops::DivAssign;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Div)
    }

    impl_op_assign!(
        (DivAssign|div_assign) => {
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
