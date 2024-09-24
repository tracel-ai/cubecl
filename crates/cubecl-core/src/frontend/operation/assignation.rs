use half::{bf16, f16};

use crate::{
    frontend::{Array, CubeContext, ExpandElement, SharedMemory, Tensor},
    prelude::{CubeIndex, CubeIndexMut, CubeType},
};
use crate::{ir, prelude::Index};

pub mod assign {
    use crate::prelude::ExpandElementTyped;

    use self::ir::{Operator, UnaryOperator};

    use super::*;

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        input: ExpandElementTyped<C>,
        output: ExpandElementTyped<C>,
    ) {
        context.register(Operator::Assign(UnaryOperator {
            input: *input.expand,
            out: *output.expand,
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

    pub fn expand<A: CubeType + CubeIndex<ExpandElementTyped<u32>>>(
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
            Variable::Local { .. } | Variable::LocalBinding { .. } => {
                binary_expand_no_vec(context, array, index, Operator::Index)
            }
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

pub mod rem_assign_array_op {
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
        array_assign_binary_op_expand(context, array, index, value, Operator::Modulo);
    }
}

pub mod bitor_assign_array_op {
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
        array_assign_binary_op_expand(context, array, index, value, Operator::BitwiseOr);
    }
}

pub mod bitand_assign_array_op {
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
        array_assign_binary_op_expand(context, array, index, value, Operator::BitwiseAnd);
    }
}

pub mod bitxor_assign_array_op {
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
        array_assign_binary_op_expand(context, array, index, value, Operator::BitwiseXor);
    }
}

pub mod shl_assign_array_op {
    use self::ir::Operator;
    use super::*;
    use crate::prelude::{array_assign_binary_op_expand, CubeType, ExpandElementTyped};

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<u32>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(context, array, index, value, Operator::ShiftLeft);
    }
}

pub mod shr_assign_array_op {
    use self::ir::Operator;
    use super::*;
    use crate::prelude::{array_assign_binary_op_expand, CubeType, ExpandElementTyped};

    pub fn expand<A: CubeType + CubeIndex<u32>>(
        context: &mut CubeContext,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<u32>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(context, array, index, value, Operator::ShiftRight);
    }
}

pub mod add_assign_op {
    use std::ops::AddAssign;

    use self::ir::Operator;
    use crate::{
        frontend::operation::base::assign_op_expand,
        prelude::{CubeType, ExpandElementTyped},
    };

    use super::*;

    pub fn expand<C: CubeType + AddAssign>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Add).into()
    }
}

pub mod sub_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Sub)
    }
}

pub mod mul_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Mul)
    }
}

pub mod div_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Div)
    }
}

pub mod rem_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Modulo)
    }
}

pub mod bitor_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::BitwiseOr)
    }
}

pub mod bitand_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::BitwiseAnd)
    }
}

pub mod bitxor_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::BitwiseXor)
    }
}

pub mod shl_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<u32>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::ShiftLeft)
    }
}

pub mod shr_assign_op {
    use self::ir::Operator;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<u32>,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::ShiftRight)
    }
}
