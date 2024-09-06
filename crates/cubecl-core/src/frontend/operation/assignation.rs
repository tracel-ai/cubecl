use half::{bf16, f16};

use crate::frontend::{Array, CubeContext, ExpandElement, SharedMemory, Tensor};
use crate::ir;

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
    use std::ops::IndexMut;

    use crate::{
        frontend::CubeType,
        prelude::{ExpandElementTyped, IndexVecMut, SliceMut},
        unexpanded,
    };

    use self::ir::{BinaryOperator, Operator, Variable};

    use super::*;

    pub fn expand<A: CubeType + IndexMut<u32>>(
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

    pub fn expand_vec<A: CubeType>(
        context: &mut CubeContext,
        vec: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A>,
    ) {
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
            out: vec.expand.into(),
        }));
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubeType, I: crate::frontend::Index> core::ops::IndexMut<I> for $type<E> {
                fn index_mut(&mut self, _index: I) -> &mut Self::Output {
                    unexpanded!()
                }
            }
        };
    }
    macro_rules! impl_index_vec {
        ($($type:ident),*) => {
            $(
                impl IndexVecMut for $type {
                    fn idx_mut(&mut self, _index: u32) -> &mut Self {
                        unexpanded!()
                    }
                }

            )*
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
    impl_index_vec!(i64, i32, f16, bf16, f32, f64, u32);

    impl<'a, E: CubeType, I: Into<u32>> core::ops::IndexMut<I> for SliceMut<'a, E> {
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
        prelude::{ExpandElementTyped, IndexVec, Slice, SliceMut},
        unexpanded,
    };

    use self::ir::{Operator, Variable};

    use super::*;

    pub fn expand<A: CubeType + core::ops::Index<u32>>(
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
            impl<E: CubeType, I: crate::frontend::Index> core::ops::Index<I> for $type<E> {
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
                impl IndexVec for $type {
                    fn idx(&self, _index: u32) -> &Self {
                        unexpanded!()
                    }
                }
            )*
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);

    impl_index_vec!(i64, i32, f16, bf16, f32, f64, u32);

    impl<'a, E: CubeType, I: Into<u32>> core::ops::Index<I> for SliceMut<'a, E> {
        type Output = E;
        fn index(&self, _index: I) -> &Self::Output {
            unexpanded!()
        }
    }

    impl<'a, E: CubeType, I: Into<u32>> core::ops::Index<I> for Slice<'a, E> {
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

    pub fn expand<A: CubeType + core::ops::Index<u32>>(
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

    pub fn expand<A: CubeType + core::ops::Index<u32>>(
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

    pub fn expand<A: CubeType + core::ops::Index<u32>>(
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

    pub fn expand<A: CubeType + core::ops::Index<u32>>(
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
