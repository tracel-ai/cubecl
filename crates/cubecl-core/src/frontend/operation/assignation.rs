use core::ops::{Index, IndexMut};

use cubecl_ir::{Operator, Scope, Variable};

use crate::{
    frontend::{Array, SharedMemory, Tensor},
    prelude::*,
};
use crate::{ir, unexpanded};

pub mod cast {
    use ir::Instruction;

    use crate::prelude::NativeExpand;

    use self::ir::UnaryOperator;

    use super::*;

    pub fn expand<From: CubeType, To: CubeType>(
        scope: &Scope,
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
    use cubecl_ir::{BinaryOperator, Memory};
    use ir::{Instruction, Operation};

    use crate::prelude::NativeExpand;

    use super::*;

    /// Expand the assign operation.
    ///
    /// If you want to assign to a manually initialized const variable, look into
    /// [`expand_no_check()`].
    pub fn expand<C: CubeType>(
        scope: &Scope,
        input: NativeExpand<C>,
        output: &mut NativeExpand<C>,
    ) {
        if output.expand.is_immutable() {
            panic!("Can't assign a value to a const variable. Try to use `RuntimeCell`.");
        }

        expand_no_check(scope, input, output);
    }
    /// Expand the assign operation without any check.
    ///
    /// You can't assign to a const variable with this [`expand()`].
    pub fn expand_no_check<C: CubeType>(
        scope: &Scope,
        input: NativeExpand<C>,
        output: &mut NativeExpand<C>,
    ) {
        let output = output.expand;
        let input = input.expand;

        expand_element(scope, input, output);
    }

    pub fn expand_element(scope: &Scope, input: Variable, output: Variable) {
        if output.ty.is_ptr() && !input.ty.is_ptr() {
            scope.register(Instruction::no_out(Memory::Store(BinaryOperator {
                lhs: output,
                rhs: input,
            })));
        } else {
            scope.register(Instruction::new(Operation::Copy(input), output));
        }
    }
}

pub mod index_mut {
    use super::*;

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> IndexMut<usize> for $type<E> {
                fn index_mut(&mut self, _idx: usize) -> &mut Self::Output {
                    unexpanded!()
                }
            }
            impl<E: CubePrimitive> IndexMutExpand<NativeExpand<usize>> for NativeExpand<$type<E>> {
                fn __expand_index_mut_method(
                    &mut self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &mut Self::Output {
                    expand_index_mut_native(scope, self, index, None, true)
                }

                fn __expand_index_mut_unchecked_method(
                    &mut self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &mut Self::Output {
                    expand_index_mut_native(scope, self, index, None, false)
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
}

pub mod index {
    use super::*;

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubePrimitive> Index<usize> for $type<E> {
                type Output = E;

                fn index(&self, _idx: usize) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexExpand<NativeExpand<usize>> for NativeExpand<$type<E>> {
                type Output = NativeExpand<E>;

                fn __expand_index_method(
                    &self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &Self::Output {
                    expand_index_native(scope, self, index, None, true)
                }
                fn __expand_index_unchecked_method(
                    &self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &Self::Output {
                    expand_index_native(scope, self, index, None, false)
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
}
