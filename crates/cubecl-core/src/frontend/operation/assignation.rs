use cubecl_ir::{Operator, Scope, Variable};

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
                input: input.expand,
            }),
            output.expand,
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
    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        input: NativeExpand<C>,
        output: &mut NativeExpand<C>,
    ) {
        let output = output.expand;
        let input = input.expand;

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
        let output = output.expand;
        let input = input.expand;

        scope.register(Instruction::new(Operation::Copy(input), output));
    }

    pub fn expand_element(scope: &mut Scope, input: Variable, output: Variable) {
        if output.is_immutable() {
            panic!("Can't assign a value to a const variable. Try to use `RuntimeCell`.");
        }

        scope.register(Instruction::new(Operation::Copy(input), output));
    }
}

pub mod index_mut {
    use super::*;

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> CubeIndexMut for $type<E> {}

            impl<E: CubePrimitive> CubeIndexMutExpand for NativeExpand<$type<E>> {
                fn __expand_index_mut_method<'a, 'b>(
                    &'a mut self,
                    scope: &'b mut Scope,
                    index: NativeExpand<usize>,
                ) -> &'a mut Self::Output {
                    expand_index_mut_native(scope, self, index, None, true)
                }
            }
        };
    }

    impl<E: Scalar, N: Size> CubeIndexMut for Vector<E, N> {}

    impl<E: Scalar, N: Size> CubeIndexMutExpand for NativeExpand<Vector<E, N>> {
        fn __expand_index_mut_method<'a, 'b>(
            &'a mut self,
            scope: &'b mut Scope,
            index: NativeExpand<usize>,
        ) -> &'a mut Self::Output {
            expand_index_mut_native::<NativeExpand<Vector<E, N>>>(scope, self, index, None, true)
        }
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
}

pub mod index {
    use super::*;

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> CubeIndex for $type<E> {
                type Output = E;
                type Idx = usize;
            }

            impl<E: CubePrimitive> CubeIndexExpand for NativeExpand<$type<E>> {
                type Output = NativeExpand<E>;
                type Idx = NativeExpand<usize>;

                fn __expand_index_method<'a, 'b>(
                    &'a self,
                    scope: &'b mut Scope,
                    index: NativeExpand<usize>,
                ) -> &'a Self::Output {
                    expand_index_native(scope, self, index, None, true)
                }
                fn __expand_index_unchecked_method<'a, 'b>(
                    &'a self,
                    scope: &'b mut Scope,
                    index: NativeExpand<usize>,
                ) -> &'a Self::Output {
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
        fn __expand_index_method<'a, 'b>(
            &'a self,
            scope: &'b mut Scope,
            index: NativeExpand<usize>,
        ) -> &'a Self::Output {
            expand_index_native(scope, self, index, None, true)
        }
        fn __expand_index_unchecked_method<'a, 'b>(
            &'a self,
            scope: &'b mut Scope,
            index: NativeExpand<usize>,
        ) -> &'a Self::Output {
            expand_index_native(scope, self, index, None, false)
        }
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
}
