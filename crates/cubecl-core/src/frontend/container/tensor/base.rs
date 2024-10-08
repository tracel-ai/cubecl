use crate::frontend::{ExpandElementBaseInit, ExpandElementTyped, SizedContainer};
use crate::{
    frontend::{indexation::Index, CubeContext, CubePrimitive, CubeType, ExpandElement},
    ir::{Elem, Item, Metadata, Variable},
    prelude::Line,
    unexpanded,
};
use std::{marker::PhantomData, num::NonZero};

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use super::*;
    use crate::prelude::Array;

    impl<T: CubeType> Tensor<T> {
        /// Obtain the stride of input at dimension dim
        pub fn stride<C: Index>(&self, _dim: C) -> u32 {
            unexpanded!()
        }

        /// Obtain the shape of input at dimension dim
        pub fn shape<C: Index>(&self, _dim: C) -> u32 {
            unexpanded!()
        }

        /// The length of the buffer representing the tensor.
        ///
        /// # Warning
        ///
        /// The length will be affected by the vectorization factor. To obtain the number of elements,
        /// you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> u32 {
            unexpanded!()
        }

        // Expand function of [stride](Tensor::stride).
        pub fn __expand_stride<C: Index>(
            context: &mut CubeContext,
            expand: ExpandElementTyped<Tensor<T>>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_stride_method(context, dim)
        }

        // Expand function of [shape](Tensor::shape).
        pub fn __expand_shape<C: Index>(
            context: &mut CubeContext,
            expand: ExpandElementTyped<Tensor<T>>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_shape_method(context, dim)
        }

        // Expand function of [len](Tensor::len).
        pub fn __expand_len<C: Index>(
            context: &mut CubeContext,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_len_method(context)
        }

        // Expand function of [rank](Tensor::rank).
        pub fn __expand_rank<C: Index>(
            context: &mut CubeContext,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_rank_method(context)
        }
    }

    impl<T: CubeType> ExpandElementTyped<Tensor<T>> {
        // Expand method of [stride](Tensor::stride).
        pub fn __expand_stride_method(
            self,
            context: &mut CubeContext,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let dim: ExpandElement = dim.into();
            let out = context.create_local_binding(Item::new(Elem::UInt));
            context.register(Metadata::Stride {
                dim: *dim,
                var: self.expand.into(),
                out: out.clone().into(),
            });
            out.into()
        }

        // Expand method of [shape](Tensor::shape).
        pub fn __expand_shape_method(
            self,
            context: &mut CubeContext,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let dim: ExpandElement = dim.into();
            let out = context.create_local_binding(Item::new(Elem::UInt));
            context.register(Metadata::Shape {
                dim: *dim,
                var: self.expand.into(),
                out: out.clone().into(),
            });
            out.into()
        }

        // Expand method of [len](Tensor::len).
        pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(context)
        }

        // Expand method of [rank](Tensor::rank).
        pub fn __expand_rank_method(self, _context: &mut CubeContext) -> ExpandElementTyped<u32> {
            ExpandElement::Plain(Variable::Rank).into()
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

    impl<E: CubePrimitive> Tensor<E> {
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

    impl<E: CubePrimitive> ExpandElementTyped<Tensor<E>> {
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

/// Module that contains the implementation details of the line_size function.
mod line {
    use super::*;

    impl<P: CubePrimitive> Tensor<Line<P>> {
        /// Get the size of each line contained in the tensor.
        ///
        /// Same as the following:
        ///
        /// ```rust, ignore
        /// let size = tensor[0].size();
        /// ```
        pub fn line_size(&self) -> u32 {
            unexpanded!()
        }

        // Expand function of [size](Tensor::line_size).
        pub fn __expand_line_size(
            expand: <Self as CubeType>::ExpandType,
            context: &mut CubeContext,
        ) -> u32 {
            expand.__expand_line_size_method(context)
        }
    }

    impl<P: CubePrimitive> ExpandElementTyped<Tensor<Line<P>>> {
        /// Comptime version of [size](Tensor::line_size).
        pub fn line_size(&self) -> u32 {
            self.expand
                .item()
                .vectorization
                .unwrap_or(NonZero::new(1).unwrap())
                .get() as u32
        }

        // Expand method of [size](Tensor::line_size).
        pub fn __expand_line_size_method(&self, _content: &mut CubeContext) -> u32 {
            self.line_size()
        }
    }
}

impl<T: CubeType<ExpandType = ExpandElementTyped<T>>> SizedContainer for Tensor<T> {
    type Item = T;
}

impl<T: CubeType> Iterator for &Tensor<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

impl<T: CubeType> CubeType for Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<C: CubeType> ExpandElementBaseInit for Tensor<C> {
    fn init_elem(_context: &mut crate::prelude::CubeContext, elem: ExpandElement) -> ExpandElement {
        // The type can't be deeply cloned/copied.
        elem
    }
}
