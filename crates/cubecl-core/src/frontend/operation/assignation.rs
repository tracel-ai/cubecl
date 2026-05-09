use core::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use cubecl_ir::{Operator, Scope, Variable};

use crate::{
    frontend::{Array, SharedMemory, Tensor},
    prelude::*,
};
use crate::{ir, unexpanded};

type ArrayExpand<E> = NativeExpand<Array<E>>;

pub mod cast {
    use ir::Instruction;

    use crate::prelude::NativeExpand;

    use self::ir::UnaryOperands;

    use super::*;

    pub fn expand<From: CubeType, To: CubeType>(
        scope: &Scope,
        input: NativeExpand<From>,
        output: NativeExpand<To>,
    ) {
        scope.register(Instruction::new(
            Operator::Cast(UnaryOperands {
                input: input.expand,
            }),
            output.expand,
        ));
    }
}

pub mod assign {
    use cubecl_ir::{Memory, StoreOperands};
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
        match (input.ty.is_ptr(), output.ty.is_ptr()) {
            (true, false) => {
                // ptr -> value = load
                scope.register(Instruction::new(Memory::Load(input), output));
            }
            (false, true) => {
                // value -> ptr = store
                scope.register(Instruction::no_out(Memory::Store(StoreOperands {
                    ptr: output,
                    value: input,
                })));
            }
            _ => {
                // same ty = copy
                scope.register(Instruction::new(Operation::Copy(input), output));
            }
        }
    }
}

pub mod index_mut {
    use super::*;

    macro_rules! impl_index {
        ($type:ident, $expand: ty) => {
            impl<E: CubePrimitive> IndexMut<usize> for $type<E> {
                fn index_mut(&mut self, _idx: usize) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<Range<usize>> for $type<E> {
                fn index_mut(&mut self, _idx: Range<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeFrom<usize>> for $type<E> {
                fn index_mut(&mut self, _idx: RangeFrom<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeFull> for $type<E> {
                fn index_mut(&mut self, _idx: RangeFull) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeInclusive<usize>> for $type<E> {
                fn index_mut(&mut self, _idx: RangeInclusive<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeTo<usize>> for $type<E> {
                fn index_mut(&mut self, _idx: RangeTo<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeToInclusive<usize>> for $type<E> {
                fn index_mut(&mut self, _idx: RangeToInclusive<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMutExpand<NativeExpand<usize>> for $expand {
                fn __expand_index_mut_method(
                    &mut self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &mut E::ExpandType {
                    self.__expand_as_mut_slice_method(scope)
                        .__expand_index_mut_method(scope, index)
                }
            }
        };
    }

    impl_index!(Array, NativeExpand<Array<E>>);
    impl_index!(Tensor, TensorExpand<E>);
    impl_index!(SharedMemory, NativeExpand<Shared<[E]>>);

    impl_slice_ranges!(ArrayExpand);
    impl_slice_ranges!(TensorExpand);
    impl_slice_ranges!(SharedMemoryExpand);
}

pub mod index {
    use super::*;

    macro_rules! impl_index {
        ($type:ident, $expand: ident) => {
            impl<E: CubePrimitive> Index<usize> for $type<E> {
                type Output = E;

                fn index(&self, _idx: usize) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<Range<usize>> for $type<E> {
                type Output = [E];

                fn index(&self, _idx: Range<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeFrom<usize>> for $type<E> {
                type Output = [E];

                fn index(&self, _idx: RangeFrom<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeFull> for $type<E> {
                type Output = [E];

                fn index(&self, _idx: RangeFull) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeInclusive<usize>> for $type<E> {
                type Output = [E];

                fn index(&self, _idx: RangeInclusive<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeTo<usize>> for $type<E> {
                type Output = [E];

                fn index(&self, _idx: RangeTo<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeToInclusive<usize>> for $type<E> {
                type Output = [E];

                fn index(&self, _idx: RangeToInclusive<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexExpand<NativeExpand<usize>> for $expand<E> {
                type Output = NativeExpand<E>;

                fn __expand_index_method(
                    &self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &Self::Output {
                    self.__expand_as_slice_method(scope)
                        .__expand_index_method(scope, index)
                }
            }
        };
    }

    impl_index!(Array, ArrayExpand);
    impl_index!(Tensor, TensorExpand);
    impl_index!(SharedMemory, SharedMemoryExpand);
}
