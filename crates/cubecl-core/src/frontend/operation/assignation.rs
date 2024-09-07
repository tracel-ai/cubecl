use half::{bf16, f16};

use crate::{
    frontend::{Array, CubeContext, ExpandElement, SharedMemory, Tensor},
    prelude::{CubeIndex, CubeIndexMut},
};
use crate::{ir, prelude::Index};

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
    };

    use self::ir::{BinaryOperator, Operator, Variable};

    use super::*;

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
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
            impl<E: CubeType, I: Index> CubeIndexMut<I> for $type<E> {}
        };
    }
    macro_rules! impl_index_vec {
        ($($type:ident),*) => {
            $(
                impl<I: Index> CubeIndexMut<I> for $type {}
            )*
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
    impl_index_vec!(i64, i32, f16, bf16, f32, f64, u32);

    impl<'a, E: CubeType, I: Index> CubeIndexMut<I> for SliceMut<'a, E> {}
}

pub mod index {
    use crate::{
        frontend::{
            operation::base::{binary_expand, binary_expand_no_vec},
            CubeType,
        },
        prelude::{ExpandElementTyped, Slice, SliceMut},
    };

    use self::ir::{Operator, Variable};

    use super::*;

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
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
            impl<E: CubeType, I: Index> CubeIndex<I> for $type<E> {
                type Output = E;
            }
        };
    }
    macro_rules! impl_index_vec {
        ($($type:ident),*) => {
            $(
                impl<I: Index> CubeIndex<I> for $type {
                    type Output = Self;
                }
            )*
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
    impl_index_vec!(i64, i32, f16, bf16, f32, f64, u32);

    impl<'a, E: CubeType, I: Index> CubeIndex<I> for Slice<'a, E> {
        type Output = E;
    }

    impl<'a, E: CubeType, I: Index> CubeIndex<I> for SliceMut<'a, E> {
        type Output = E;
    }
}

pub mod add_assign_array_op {
    use self::ir::Operator;
    use super::*;
    use crate::prelude::{array_assign_binary_op_expand, CubeType, ExpandElementTyped};

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
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

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
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

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
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

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(context, array, index, value, Operator::Div);
    }
}

pub mod add_assign_op {
    use self::ir::Operator;
    use crate::frontend::operation::base::assign_op_expand;

    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Add)
    }
}

pub mod sub_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::frontend::operation::base::assign_op_expand;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Sub)
    }
}

pub mod mul_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::frontend::operation::base::assign_op_expand;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Mul)
    }
}

pub mod div_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::frontend::operation::base::assign_op_expand;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Div)
    }
}
