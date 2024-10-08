use std::{marker::PhantomData, num::NonZero};

use crate::frontend::{
    CubePrimitive, ExpandElement, ExpandElementBaseInit, ExpandElementTyped, IntoRuntime,
};
use crate::prelude::SizedContainer;
use crate::{
    frontend::CubeType,
    ir::{Elem, Item, Metadata},
    unexpanded,
};
use crate::{
    frontend::{indexation::Index, CubeContext},
    prelude::{assign, index, index_assign},
};

/// A contiguous array of elements.
pub struct Array<E> {
    _val: PhantomData<E>,
}

/// Module that contains the implementation details of the new function.
mod new {
    use super::*;
    use crate::ir::Variable;

    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create a new array of the given length.
        #[allow(unused_variables)]
        pub fn new<L: Index>(length: L) -> Self {
            Array { _val: PhantomData }
        }

        /// Create an array from data.
        pub fn from_data<C: CubePrimitive>(_data: impl IntoIterator<Item = C>) -> Self {
            Array { _val: PhantomData }
        }

        /// Expand function of [new](Array::new).
        pub fn __expand_new(
            context: &mut CubeContext,
            size: ExpandElementTyped<u32>,
        ) -> <Self as CubeType>::ExpandType {
            let size = size
                .constant()
                .expect("Array need constant initialization value")
                .as_u32();
            context
                .create_local_array(Item::new(T::as_elem()), size)
                .into()
        }

        /// Expand function of [from_data](Array::from_data).
        pub fn __expand_from_data<C: CubePrimitive>(
            context: &mut CubeContext,
            data: ArrayData<C>,
        ) -> <Self as CubeType>::ExpandType {
            let var = context.create_const_array(Item::new(T::as_elem()), data.values);
            ExpandElementTyped::new(var)
        }
    }

    /// Type useful for the expand function of [from_data](Array::from_data).
    pub struct ArrayData<C> {
        values: Vec<Variable>,
        _ty: PhantomData<C>,
    }

    impl<C: CubePrimitive + Into<ExpandElementTyped<C>>, T: IntoIterator<Item = C>> From<T>
        for ArrayData<C>
    {
        fn from(value: T) -> Self {
            let values: Vec<Variable> = value
                .into_iter()
                .map(|value| {
                    let value: ExpandElementTyped<C> = value.into();
                    *value.expand
                })
                .collect();
            ArrayData {
                values,
                _ty: PhantomData,
            }
        }
    }
}

/// Module that contains the implementation details of vectorization functions.
///
/// TODO: Remove vectorization in favor of the line API.
mod vectorization {
    use super::*;

    impl<T: CubePrimitive + Clone> Array<T> {
        #[allow(unused_variables)]
        pub fn vectorized<L: Index>(length: L, vectorization_factor: u32) -> Self {
            Array { _val: PhantomData }
        }

        pub fn to_vectorized(self, _vectorization_factor: u32) -> T {
            unexpanded!()
        }

        pub fn __expand_vectorized(
            context: &mut CubeContext,
            size: ExpandElementTyped<u32>,
            vectorization_factor: u32,
        ) -> <Self as CubeType>::ExpandType {
            let size = size.value();
            let size = match size {
                crate::ir::Variable::ConstantScalar(value) => value.as_u32(),
                _ => panic!("Shared memory need constant initialization value"),
            };
            context
                .create_local_array(
                    Item::vectorized(T::as_elem(), NonZero::new(vectorization_factor as u8)),
                    size,
                )
                .into()
        }
    }

    impl<C: CubePrimitive> ExpandElementTyped<Array<C>> {
        pub fn __expand_to_vectorized_method(
            self,
            context: &mut CubeContext,
            vectorization_factor: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<C> {
            let factor = vectorization_factor
                .constant()
                .expect("Vectorization must be comptime")
                .as_u32();
            let var = self.expand.clone();
            let item = Item::vectorized(var.item().elem(), NonZero::new(factor as u8));

            let new_var = if factor == 1 {
                let new_var = context.create_local_binding(item);
                let element =
                    index::expand(context, self.clone(), ExpandElementTyped::from_lit(0u32));
                assign::expand(context, element, new_var.clone().into());
                new_var
            } else {
                let new_var = context.create_local_variable(item);
                for i in 0..factor {
                    let expand: Self = self.expand.clone().into();
                    let element = index::expand(context, expand, ExpandElementTyped::from_lit(i));
                    index_assign::expand::<Array<C>>(
                        context,
                        new_var.clone().into(),
                        ExpandElementTyped::from_lit(i),
                        element,
                    );
                }
                new_var
            };
            new_var.into()
        }
    }
}

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use super::*;

    impl<E: CubeType> Array<E> {
        /// Obtain the array length
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }
    }

    impl<T: CubeType> ExpandElementTyped<Array<T>> {
        // Expand method of [len](Array::len).
        pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
            let out = context.create_local_binding(Item::new(Elem::UInt));
            context.register(Metadata::Length {
                var: self.expand.into(),
                out: out.clone().into(),
            });
            out.into()
        }
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use crate::{
        ir::{BinaryOperator, Operator},
        prelude::{CubeIndex, CubeIndexMut},
    };

    use super::*;

    impl<E: CubePrimitive> Array<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_unchecked<I: Index>(&self, _i: I) -> &E
        where
            Self: CubeIndex<I>,
        {
            unexpanded!()
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_assign_unchecked<I: Index>(&mut self, _i: I, _value: E)
        where
            Self: CubeIndexMut<I>,
        {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<Array<E>> {
        pub fn __expand_index_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = context.create_local_binding(self.expand.item());
            context.register(Operator::UncheckedIndex(BinaryOperator {
                out: *out,
                lhs: *self.expand,
                rhs: i.expand.consume(),
            }));
            out.into()
        }

        pub fn __expand_index_assign_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
            value: ExpandElementTyped<E>,
        ) {
            context.register(Operator::UncheckedIndexAssign(BinaryOperator {
                out: *self.expand,
                lhs: i.expand.consume(),
                rhs: value.expand.consume(),
            }));
        }
    }
}

impl<E: CubePrimitive> IntoRuntime for Array<E> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        unimplemented!("Array can't exist at compile time")
    }
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> CubeType for &Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> ExpandElementBaseInit for Array<C> {
    fn init_elem(_context: &mut crate::prelude::CubeContext, elem: ExpandElement) -> ExpandElement {
        // The type can't be deeply cloned/copied.
        elem
    }
}

impl<T: CubeType<ExpandType = ExpandElementTyped<T>>> SizedContainer for Array<T> {
    type Item = T;
}

impl<T: CubeType> Iterator for &Array<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}
