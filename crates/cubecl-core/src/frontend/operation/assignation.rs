use cubecl_ir::{Bitwise, ManagedVariable, Operator, Scope};

use crate::ir;
use crate::{
    frontend::{Array, SharedMemory, Tensor},
    prelude::*,
};

pub mod cast {
    use ir::Instruction;

    use crate::prelude::NativeExpand;

    use self::ir::UnaryOperator;

    use super::*;

    pub fn expand<From: CubeType, To: CubeType>(
        scope: &mut Scope,
        input: NativeExpand<From>,
        output: NativeExpand<To>,
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

    use crate::prelude::NativeExpand;

    use super::*;

    /// Expand the assign operation.
    ///
    /// If you want to assign to a manually initialized const variable, look into
    /// [`expand_no_check()`].
    pub fn expand<C: CubeType>(scope: &mut Scope, input: NativeExpand<C>, output: NativeExpand<C>) {
        let output = *output.expand;
        let input = *input.expand;

        if output.is_immutable() {
            panic!("Can't assign a value to a const variable. Try to use `RuntimeCell`.");
        }

        scope.register(Instruction::new(Operation::Copy(input), output));
    }
    /// Expand the assign operation without any check.
    ///
    /// You can't assign to a const variable with this [`expand()`].
    pub fn expand_no_check<C: CubeType>(
        scope: &mut Scope,
        input: NativeExpand<C>,
        output: NativeExpand<C>,
    ) {
        let output = *output.expand;
        let input = *input.expand;

        scope.register(Instruction::new(Operation::Copy(input), output));
    }

    pub fn expand_element(scope: &mut Scope, input: ManagedVariable, output: ManagedVariable) {
        if output.is_immutable() {
            panic!("Can't assign a value to a const variable. Try to use `RuntimeCell`.");
        }

        scope.register(Instruction::new(Operation::Copy(*input), *output));
    }
}

pub mod index_assign {
    use super::*;

    pub fn expand<A: CubeIndexMutExpand<Output = NativeExpand<V>>, V: CubePrimitive>(
        scope: &mut Scope,
        expand: A,
        index: A::Idx,
        value: NativeExpand<V>,
    ) {
        expand.expand_index_mut(scope, index, value)
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> CubeIndexMut for $type<E> {}

            impl<E: CubePrimitive> CubeIndexMutExpand for NativeExpand<$type<E>> {
                fn expand_index_mut(
                    self,
                    scope: &mut Scope,
                    index: NativeExpand<usize>,
                    value: Self::Output,
                ) {
                    expand_index_assign_native::<$type<E>>(scope, self, index, value, None, true);
                }
            }
        };
    }

    impl<E: Scalar, N: Size> CubeIndexMut for Vector<E, N> {}

    impl<E: Scalar, N: Size> CubeIndexMutExpand for NativeExpand<Vector<E, N>> {
        fn expand_index_mut(
            self,
            scope: &mut Scope,
            index: NativeExpand<usize>,
            value: Self::Output,
        ) {
            expand_index_assign_native::<Vector<E, N>>(scope, self, index, value, None, true);
        }
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
}

pub mod index {
    use super::*;

    pub fn expand<A: CubeIndexExpand<Output = NativeExpand<V>>, V: CubeType>(
        scope: &mut Scope,
        expand: A,
        index: A::Idx,
    ) -> NativeExpand<V> {
        expand.expand_index(scope, index)
    }

    pub fn expand_with<A: CubeIndexExpand<Output = NativeExpand<V>>, V: CubeType>(
        scope: &mut Scope,
        expand: A,
        index: A::Idx,
    ) -> NativeExpand<V> {
        expand.expand_index(scope, index)
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> CubeIndex for $type<E> {
                type Output = E;
                type Idx = usize;
            }

            impl<E: CubePrimitive> CubeIndexExpand for NativeExpand<$type<E>> {
                type Output = NativeExpand<E>;
                type Idx = NativeExpand<usize>;

                fn expand_index(
                    self,
                    scope: &mut Scope,
                    index: NativeExpand<usize>,
                ) -> Self::Output {
                    expand_index_native(scope, self, index, None, true)
                }
                fn expand_index_unchecked(
                    self,
                    scope: &mut Scope,
                    index: NativeExpand<usize>,
                ) -> Self::Output {
                    expand_index_native(scope, self, index, None, false)
                }
            }
        };
    }

    impl<E: Scalar, N: Size> CubeIndex for Vector<E, N> {
        type Output = E;
        type Idx = usize;
    }
    impl<E: Scalar, N: Size> CubeIndexExpand for NativeExpand<Vector<E, N>> {
        type Output = NativeExpand<E>;
        type Idx = NativeExpand<usize>;
        fn expand_index(self, scope: &mut Scope, index: NativeExpand<usize>) -> Self::Output {
            expand_index_native(scope, self, index, None, true)
        }
        fn expand_index_unchecked(
            self,
            scope: &mut Scope,
            index: NativeExpand<usize>,
        ) -> Self::Output {
            expand_index_native(scope, self, index, None, false)
        }
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
}

pub mod index_unchecked {
    use super::*;
    use crate::prelude::{CubeIndexExpand, NativeExpand};

    pub fn expand<A: CubeIndexExpand<Output = NativeExpand<V>>, V: CubeType>(
        scope: &mut Scope,
        expand: A,
        index: A::Idx,
    ) -> NativeExpand<V> {
        expand.expand_index_unchecked(scope, index)
    }
}

pub mod add_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Add);
    }
}

pub mod sub_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Sub);
    }
}

pub mod mul_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Mul);
    }
}

pub mod div_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Div);
    }
}

pub mod rem_assign_array_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Arithmetic::Modulo);
    }
}

pub mod bitor_assign_array_op {
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::BitwiseOr);
    }
}

pub mod bitand_assign_array_op {
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::BitwiseAnd);
    }
}

pub mod bitxor_assign_array_op {
    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<A::Output>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::BitwiseXor);
    }
}

pub mod shl_assign_array_op {

    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<u32>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::ShiftLeft);
    }
}

pub mod shr_assign_array_op {

    use super::*;
    use crate::prelude::{CubeType, NativeExpand, array_assign_binary_op_expand};

    pub fn expand<A: CubeType + CubeIndex>(
        scope: &mut Scope,
        array: NativeExpand<A>,
        index: NativeExpand<usize>,
        value: NativeExpand<u32>,
    ) where
        A::Output: CubeType + Sized,
    {
        array_assign_binary_op_expand(scope, array, index, value, Bitwise::ShiftRight);
    }
}

pub mod add_assign_op {
    use core::ops::AddAssign;

    use self::ir::Arithmetic;
    use crate::{
        frontend::operation::base::assign_op_expand,
        prelude::{CubeType, NativeExpand},
    };

    use super::*;

    pub fn expand<C: CubeType + AddAssign>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Add).into()
    }
}

pub mod sub_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Sub)
    }
}

pub mod mul_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Mul)
    }
}

pub mod div_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Div)
    }
}

pub mod rem_assign_op {
    use self::ir::Arithmetic;
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Modulo)
    }
}

pub mod bitor_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseOr)
    }
}

pub mod bitand_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseAnd)
    }
}

pub mod bitxor_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::BitwiseXor)
    }
}

pub mod shl_assign_op {

    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<u32>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::ShiftLeft)
    }
}

pub mod shr_assign_op {
    use super::*;
    use crate::{frontend::operation::base::assign_op_expand, prelude::NativeExpand};

    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<u32>,
    ) -> ManagedVariable {
        assign_op_expand(scope, lhs.into(), rhs.into(), Bitwise::ShiftRight)
    }
}

pub mod add_assign {
    use cubecl_ir::Arithmetic;

    use super::*;
    use crate::prelude::{CubePrimitive, NativeExpand, assign_op_expand};

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<C> {
        assign_op_expand(scope, lhs.into(), rhs.into(), Arithmetic::Add).into()
    }
}
