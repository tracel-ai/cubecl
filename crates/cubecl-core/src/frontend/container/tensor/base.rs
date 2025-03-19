use crate::{
    frontend::{
        CubePrimitive, CubeType, ExpandElementBaseInit, ExpandElementTyped, SizedContainer,
        indexation::Index,
    },
    ir::{Item, Metadata, Scope},
    prelude::{Line, List, ListExpand, ListMut, ListMutExpand, index, index_assign},
    unexpanded,
};
use cubecl_ir::ExpandElement;
use std::{marker::PhantomData, num::NonZero};

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use cubecl_ir::ExpandElement;

    use super::*;
    use crate::{
        ir::{Arithmetic, BinaryOperator, Instruction},
        prelude::Array,
    };

    impl<T: CubeType> Tensor<T> {
        /// Obtain the stride of input at dimension dim
        pub fn stride<C: Index>(&self, _dim: C) -> u32 {
            unexpanded!()
        }

        /// Obtain the shape of input at dimension dim
        pub fn shape<C: Index>(&self, _dim: C) -> u32 {
            unexpanded!()
        }

        /// Obtain the coordinate corresponding to the given `index` of the tensor at dimension `dim`.
        ///
        /// A coordinate is a list of indices corresponding to the multi-dimensional position of an element in the tensor.
        /// The `dim` element in a coordinate is the position along the `dim` dimension of the tensor.
        pub fn coordinate<I: Index, D: Index>(&self, _index: I, _dim: D) -> u32 {
            unexpanded!()
        }

        /// The number of vectorized elements in the tensor.
        ///
        /// # Warning
        ///
        /// The length will be affected by the vectorization factor. To obtain the number of elements,
        /// you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }

        /// The length of the buffer representing the tensor in terms of vectorized elements.
        ///
        /// # Warning
        ///
        /// The buffer length will be affected by the vectorization factor. To obtain the number of
        /// elements, you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn buffer_len(&self) -> u32 {
            unexpanded!()
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> u32 {
            unexpanded!()
        }

        // Expand function of [stride](Tensor::stride).
        pub fn __expand_stride<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_stride_method(scope, dim)
        }

        // Expand function of [shape](Tensor::shape).
        pub fn __expand_shape<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_shape_method(scope, dim)
        }

        // Expand function of [coordinate](Tensor::coordinate).
        pub fn __expand_coordinate<I: Index, D: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
            index: ExpandElementTyped<u32>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_coordinate_method(scope, index, dim)
        }

        // Expand function of [len](Tensor::len).
        pub fn __expand_len<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_len_method(scope)
        }

        // Expand function of [buffer_len](Tensor::buffer_len).
        pub fn __expand_buffer_len<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_buffer_len_method(scope)
        }

        // Expand function of [rank](Tensor::rank).
        pub fn __expand_rank<C: Index>(
            scope: &mut Scope,
            expand: ExpandElementTyped<Tensor<T>>,
        ) -> ExpandElementTyped<u32> {
            expand.__expand_rank_method(scope)
        }
    }

    impl<T: CubeType> ExpandElementTyped<Tensor<T>> {
        // Expand method of [stride](Tensor::stride).
        pub fn __expand_stride_method(
            self,
            scope: &mut Scope,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let dim: ExpandElement = dim.into();
            let out = scope.create_local(Item::new(u32::as_elem(scope)));
            scope.register(Instruction::new(
                Metadata::Stride {
                    dim: *dim,
                    var: self.expand.into(),
                },
                out.clone().into(),
            ));
            out.into()
        }

        // Expand method of [shape](Tensor::shape).
        pub fn __expand_shape_method(
            self,
            scope: &mut Scope,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let dim: ExpandElement = dim.into();
            let out = scope.create_local(Item::new(u32::as_elem(scope)));
            scope.register(Instruction::new(
                Metadata::Shape {
                    dim: *dim,
                    var: self.expand.into(),
                },
                out.clone().into(),
            ));
            out.into()
        }

        // Expand method of [coordinate](Tensor::coordinate).
        pub fn __expand_coordinate_method(
            self,
            scope: &mut Scope,
            index: ExpandElementTyped<u32>,
            dim: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            let index: ExpandElement = index.into();
            let stride = self.clone().__expand_stride_method(scope, dim.clone());
            let shape = self.clone().__expand_shape_method(scope, dim.clone());

            // Compute `num_strides = index / stride`.
            let num_strides = scope.create_local(Item::new(u32::as_elem(scope)));
            scope.register(Instruction::new(
                Arithmetic::Div(BinaryOperator {
                    lhs: *index,
                    rhs: stride.expand.into(),
                }),
                num_strides.clone().into(),
            ));

            // Compute `coordinate = num_strides % shape `.
            let coordinate = scope.create_local(Item::new(u32::as_elem(scope)));
            scope.register(Instruction::new(
                Arithmetic::Modulo(BinaryOperator {
                    lhs: *num_strides,
                    rhs: shape.expand.into(),
                }),
                coordinate.clone().into(),
            ));

            coordinate.into()
        }

        // Expand method of [len](Tensor::len).
        pub fn __expand_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(scope)
        }

        // Expand method of [buffer_len](Tensor::buffer_len).
        pub fn __expand_buffer_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_buffer_len_method(scope)
        }

        // Expand method of [rank](Tensor::rank).
        pub fn __expand_rank_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            let out = scope.create_local(Item::new(u32::as_elem(scope)));
            scope.register(Instruction::new(Metadata::Rank { var: *self.expand }, *out));
            out.into()
        }
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::Operator;

    use crate::{
        ir::{BinaryOperator, Instruction},
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
            scope: &mut Scope,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = scope.create_local(self.expand.item);
            scope.register(Instruction::new(
                Operator::UncheckedIndex(BinaryOperator {
                    lhs: *self.expand,
                    rhs: i.expand.consume(),
                }),
                *out,
            ));
            out.into()
        }

        pub fn __expand_index_assign_unchecked_method(
            self,
            scope: &mut Scope,
            i: ExpandElementTyped<u32>,
            value: ExpandElementTyped<E>,
        ) {
            scope.register(Instruction::new(
                Operator::UncheckedIndexAssign(BinaryOperator {
                    lhs: i.expand.consume(),
                    rhs: value.expand.consume(),
                }),
                *self.expand,
            ));
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
            scope: &mut Scope,
        ) -> u32 {
            expand.__expand_line_size_method(scope)
        }
    }

    impl<P: CubePrimitive> ExpandElementTyped<Tensor<Line<P>>> {
        /// Comptime version of [size](Tensor::line_size).
        pub fn line_size(&self) -> u32 {
            self.expand
                .item
                .vectorization
                .unwrap_or(NonZero::new(1).unwrap())
                .get() as u32
        }

        // Expand method of [size](Tensor::line_size).
        pub fn __expand_line_size_method(&self, _content: &mut Scope) -> u32 {
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

impl<T: CubeType> CubeType for *const Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<T: CubeType> CubeType for *mut Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<C: CubeType> ExpandElementBaseInit for Tensor<C> {
    fn init_elem(_scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        // The type can't be deeply cloned/copied.
        elem
    }
}

impl<T: CubePrimitive> List<T> for Tensor<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: ExpandElementTyped<Tensor<T>>,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, this, idx)
    }
}

impl<T: CubePrimitive> ListExpand<T> for ExpandElementTyped<Tensor<T>> {
    fn __expand_read_method(
        self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self, idx)
    }
}

impl<T: CubePrimitive> ListMut<T> for Tensor<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: ExpandElementTyped<Tensor<T>>,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for ExpandElementTyped<Tensor<T>> {
    fn __expand_write_method(
        self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self, idx, value);
    }
}
