use core::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use cubecl_ir::{ExpandValue, Scope};

use crate::unexpanded;
use crate::{
    frontend::{Array, Tensor},
    prelude::*,
};

type ArrayExpand<E> = NativeExpand<Array<E>>;

pub mod assign {
    use cubecl_ir::{dialect::memory::StoreOp, interfaces::TypedExt, pliron::r#type::Typed};

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
        if output.value(scope).is_immutable(scope.ctx()) {
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

    pub fn expand_element(scope: &Scope, input: ExpandValue, output: ExpandValue) {
        let mut input = input.read_value(scope);
        let output = output.value(scope);

        if output.vector_size(scope.ctx()) > 1 && input.vector_size(scope.ctx()) == 1 {
            input = cast_value(scope, input, output.get_type(scope.ctx()));
        }

        // value -> ptr = store
        let store = StoreOp::new(scope.ctx_mut(), output, input);
        scope.register(&store);
    }
}

pub mod index_mut {
    use super::*;

    macro_rules! impl_index {
        ($type: ty, $expand: ty) => {
            impl<E: CubePrimitive> IndexMut<usize> for $type {
                fn index_mut(&mut self, _idx: usize) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<Range<usize>> for $type {
                fn index_mut(&mut self, _idx: Range<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeFrom<usize>> for $type {
                fn index_mut(&mut self, _idx: RangeFrom<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeFull> for $type {
                fn index_mut(&mut self, _idx: RangeFull) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeInclusive<usize>> for $type {
                fn index_mut(&mut self, _idx: RangeInclusive<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeTo<usize>> for $type {
                fn index_mut(&mut self, _idx: RangeTo<usize>) -> &mut Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexMut<RangeToInclusive<usize>> for $type {
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

    impl_index!(Array<E>, NativeExpand<Array<E>>);
    impl_index!(Tensor<E>, TensorExpand<E>);
    impl_index!(Shared<[E]>, NativeExpand<Shared<[E]>>);

    impl_slice_ranges!(ArrayExpand<E>);
    impl_slice_ranges!(TensorExpand<E>);
    impl_slice_ranges!(NativeExpand<Shared<[E]>>);
}

pub mod index {
    use super::*;

    macro_rules! impl_index {
        ($type: ty, $expand: ty) => {
            impl<E: CubePrimitive> Index<usize> for $type {
                type Output = E;

                fn index(&self, _idx: usize) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<Range<usize>> for $type {
                type Output = [E];

                fn index(&self, _idx: Range<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeFrom<usize>> for $type {
                type Output = [E];

                fn index(&self, _idx: RangeFrom<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeFull> for $type {
                type Output = [E];

                fn index(&self, _idx: RangeFull) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeInclusive<usize>> for $type {
                type Output = [E];

                fn index(&self, _idx: RangeInclusive<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeTo<usize>> for $type {
                type Output = [E];

                fn index(&self, _idx: RangeTo<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> Index<RangeToInclusive<usize>> for $type {
                type Output = [E];

                fn index(&self, _idx: RangeToInclusive<usize>) -> &Self::Output {
                    unexpanded!()
                }
            }

            impl<E: CubePrimitive> IndexExpand<NativeExpand<usize>> for $expand {
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

    impl_index!(Array<E>, NativeExpand<Array<E>>);
    impl_index!(Tensor<E>, TensorExpand<E>);
    impl_index!(Shared<[E]>, NativeExpand<Shared<[E]>>);
}
