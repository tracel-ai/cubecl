use cubecl_ir::{Bitwise, ExpandElement, Operator, Scope};

use crate::ir;
use crate::{
    frontend::{Array, SharedMemory, Tensor},
    prelude::{CubeIndex, CubeIndexMut, CubePrimitive, CubeType},
};

pub mod cast {
    use ir::Instruction;

    use crate::prelude::ExpandElementTyped;

    use self::ir::UnaryOperator;

    use super::*;

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        input: ExpandElementTyped<C>,
        output: ExpandElementTyped<C>,
    ) {
        scope.register(Instruction::new(
            Operator::Cast(UnaryOperator {
                input: *input.expand,
            }),
            *output.expand,
        ));
    }
}

pub mod assign {
    use ir::{Instruction, Operation};

    use crate::prelude::ExpandElementTyped;

    use super::*;

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        input: ExpandElementTyped<C>,
        output: ExpandElementTyped<C>,
    ) {
        scope.register(Instruction::new(
            Operation::Copy(*input.expand),
            *output.expand,
        ));
    }
}

pub mod index_assign {
    use super::*;
    use crate::prelude::{
        CubeIndexMutExpand, ExpandElementTyped, Line, expand_index_assign_native,
    };

    pub fn expand<A: CubeIndexMutExpand<Output = ExpandElementTyped<V>>, V: CubePrimitive>(
        scope: &mut Scope,
        expand: A,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<V>,
    ) {
        expand.expand_index_mut(scope, index, value)
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> CubeIndexMut for $type<E> {}

            impl<E: CubePrimitive> CubeIndexMutExpand for ExpandElementTyped<$type<E>> {
                type Output = ExpandElementTyped<E>;

                fn expand_index_mut(
                    self,
                    scope: &mut Scope,
                    index: ExpandElementTyped<u32>,
                    value: Self::Output,
                ) {
                    expand_index_assign_native::<$type<E>>(scope, self, index, value);
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
    impl_index!(Line);
}

pub mod index {
    use super::*;
    use crate::prelude::{CubeIndexExpand, ExpandElementTyped, Line, expand_index_native};

    pub fn expand<A: CubeIndexExpand<Output = ExpandElementTyped<V>>, V: CubeType>(
        scope: &mut Scope,
        expand: A,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<V> {
        expand.expand_index(scope, index)
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> CubeIndex for $type<E> {
                type Output = E;
            }

            impl<E: CubePrimitive> CubeIndexExpand for ExpandElementTyped<$type<E>> {
                type Output = ExpandElementTyped<E>;

                fn expand_index(
                    self,
                    scope: &mut Scope,
                    index: ExpandElementTyped<u32>,
                ) -> Self::Output {
                    expand_index_native(scope, self, index)
                }
                fn expand_index_unchecked(
                    self,
                    scope: &mut Scope,
                    index: ExpandElementTyped<u32>,
                ) -> Self::Output {
                    expand_index_native(scope, self, index)
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
    impl_index!(Line);
}

pub mod index_unchecked {
    use super::*;
    use crate::prelude::{CubeIndexExpand, ExpandElementTyped};

    pub fn expand<A: CubeIndexExpand<Output = ExpandElementTyped<V>>, V: CubeType>(
        scope: &mut Scope,
        expand: A,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<V> {
        expand.expand_index_unchecked(scope, index)
    }
}

pub mod add_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Add);
    }
}

pub mod sub_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Sub);
    }
}

pub mod mul_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Mul);
    }
}

pub mod div_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Div);
    }
}

pub mod rem_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Modulo);
    }
}

pub mod bitor_assign_array_op {
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::BitwiseOr);
    }
}

pub mod bitand_assign_array_op {
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::BitwiseAnd);
    }
}

pub mod bitxor_assign_array_op {
    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::BitwiseXor);
    }
}

pub mod shl_assign_array_op {

    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<u32>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::ShiftLeft);
    }
}

pub mod shr_assign_array_op {

    use super::*;
    use crate::prelude::{CubeType, ExpandElementTyped, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: ExpandElementTyped<A>,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<u32>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::ShiftRight);
    }
}

pub mod add_assign_op {
    use std::ops::AddAssign;

    use self::ir::Arithmetic;
    use crate::{
        frontend::operation::base::assign_op_expand,
        prelude::{CubeType, ExpandElementTyped},
    };

    use super::*;

    pub fn expand<C: CubeType + AddAssign>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Add).into()
    }
}

pub mod sub_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Sub)
    }
}

pub mod mul_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Mul)
    }
}

pub mod div_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Div)
    }
}

pub mod rem_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Modulo)
    }
}

pub mod bitor_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseOr)
    }
}

pub mod bitand_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseAnd)
    }
}

pub mod bitxor_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseXor)
    }
}

pub mod shl_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<u32>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::ShiftLeft)
    }
}

pub mod shr_assign_op {
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::ExpandElementTyped};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<u32>,
    ) -> ExpandElement {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::ShiftRight)
    }
}

pub mod add_assign {
    use cubecl_ir::Arithmetic;

    use super::*;
    use crate::prelude::{CubePrimitive, ExpandElementTyped, assign_op_expand};

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Add).into()
    }
}
