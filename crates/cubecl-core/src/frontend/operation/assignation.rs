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
type TensorExpand<E> = NativeExpand<Tensor<E>>;

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
    use cubecl_ir::{Memory, StoreOperator};
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
                scope.register(Instruction::no_out(Memory::Store(StoreOperator {
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
        ($type:ident) => {
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

            impl<E: CubePrimitive> IndexMutExpand<NativeExpand<usize>> for NativeExpand<$type<E>> {
                fn __expand_index_mut_method(
                    &mut self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &mut E::ExpandType {
                    expand_index_mut_native(scope, self, index, None, true)
                }
            }

            impl<E: CubePrimitive> $type<E> {
                /// Returns a mutable reference to an element or subslice, without doing
                /// bounds checking.
                ///
                /// For a safe alternative see [`get_mut`].
                ///
                /// # Safety
                ///
                /// Calling this method with an out-of-bounds index is *[undefined behavior]*
                /// even if the resulting reference is not used.
                ///
                /// You can think of this like `.get_mut(index).unwrap_unchecked()`.  It's
                /// UB to call `.get_unchecked_mut(len)`, even if you immediately convert
                /// to a pointer.  And it's UB to call `.get_unchecked_mut(..len + 1)`,
                /// `.get_unchecked_mut(..=len)`, or similar.
                ///
                /// [`get_mut`]: slice::get_mut
                /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
                ///
                /// # Examples
                ///
                /// ```
                /// let x = &mut [1, 2, 4];
                ///
                /// unsafe {
                ///     let elem = x.get_unchecked_mut(1);
                ///     *elem = 13;
                /// }
                /// assert_eq!(x, &[1, 13, 4]);
                /// ```
                #[allow(unused)]
                pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut E {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> NativeExpand<$type<E>> {
                #[doc(hidden)]
                pub unsafe fn __expand_get_unchecked_mut_method(
                    &mut self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &mut NativeExpand<E> {
                    expand_index_mut_native(scope, self, index, None, false)
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);

    impl_slice_ranges!(ArrayExpand);
    impl_slice_ranges!(TensorExpand);
    impl_slice_ranges!(SharedMemoryExpand);
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

            impl<E: CubePrimitive> IndexExpand<NativeExpand<usize>> for NativeExpand<$type<E>> {
                type Output = NativeExpand<E>;

                fn __expand_index_method(
                    &self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &Self::Output {
                    expand_index_native(scope, self, index, None, true)
                }
            }

            impl<E: CubePrimitive> $type<E> {
                /// Returns a reference to an element or subslice, without doing bounds
                /// checking.
                ///
                /// For a safe alternative see [`get`].
                ///
                /// # Safety
                ///
                /// Calling this method with an out-of-bounds index is *[undefined behavior]*
                /// even if the resulting reference is not used.
                ///
                /// You can think of this like `.get(index).unwrap_unchecked()`.  It's UB
                /// to call `.get_unchecked(len)`, even if you immediately convert to a
                /// pointer.  And it's UB to call `.get_unchecked(..len + 1)`,
                /// `.get_unchecked(..=len)`, or similar.
                ///
                /// [`get`]: slice::get
                /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
                ///
                /// # Examples
                ///
                /// ```
                /// let x = &[1, 2, 4];
                ///
                /// unsafe {
                ///     assert_eq!(x.get_unchecked(1), &2);
                /// }
                /// ```
                #[allow(unused)]
                pub unsafe fn get_unchecked(&self, index: usize) -> &E {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> NativeExpand<$type<E>> {
                #[doc(hidden)]
                pub unsafe fn __expand_get_unchecked_method(
                    &self,
                    scope: &Scope,
                    index: NativeExpand<usize>,
                ) -> &NativeExpand<E> {
                    expand_index_native(scope, self, index, None, false)
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
    impl_index!(SharedMemory);
}
