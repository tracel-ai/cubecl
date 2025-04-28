use crate::{
    frontend::{
        CubePrimitive, CubeType, ExpandElementBaseInit, ExpandElementTyped, SizedContainer,
    },
    ir::{Item, Metadata, Scope},
    prelude::{
        Line, List, ListExpand, ListMut, ListMutExpand, index, index_assign, index_unchecked,
    },
    unexpanded,
};
use cubecl_ir::ExpandElement;
use cubecl_macros::{cube, intrinsic};
use std::{marker::PhantomData, num::NonZero};

use crate as cubecl;

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

type TensorExpand<T> = ExpandElementTyped<Tensor<T>>;

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use cubecl_ir::ExpandElement;

    use super::*;
    use crate::{
        ir::{Arithmetic, BinaryOperator, Instruction},
        prelude::Array,
    };

    #[cube]
    impl<T: CubeType> Tensor<T> {
        /// Obtain the stride of input at dimension dim
        #[allow(unused_variables)]
        pub fn stride(&self, dim: u32) -> u32 {
            intrinsic!(|scope| {
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
            })
        }

        /// Obtain the shape of input at dimension dim
        #[allow(unused_variables)]
        pub fn shape(&self, dim: u32) -> u32 {
            intrinsic!(|scope| {
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
            })
        }

        /// Obtain the coordinate corresponding to the given `index` of the tensor at dimension `dim`.
        ///
        /// A coordinate is a list of indices corresponding to the multi-dimensional position of an element in the tensor.
        /// The `dim` element in a coordinate is the position along the `dim` dimension of the tensor.
        #[allow(unused_variables)]
        pub fn coordinate(&self, index: u32, dim: u32) -> u32 {
            intrinsic!(|scope| {
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
            })
        }

        /// The number of vectorized elements in the tensor.
        ///
        /// # Warning
        ///
        /// The length will be affected by the vectorization factor. To obtain the number of elements,
        /// you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            intrinsic!(|scope| {
                let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
                elem.__expand_len_method(scope)
            })
        }

        /// The length of the buffer representing the tensor in terms of vectorized elements.
        ///
        /// # Warning
        ///
        /// The buffer length will be affected by the vectorization factor. To obtain the number of
        /// elements, you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn buffer_len(&self) -> u32 {
            intrinsic!(|scope| {
                let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
                elem.__expand_buffer_len_method(scope)
            })
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> u32 {
            intrinsic!(|scope| {
                let out = scope.create_local(Item::new(u32::as_elem(scope)));
                scope.register(Instruction::new(Metadata::Rank { var: *self.expand }, *out));
                out.into()
            })
        }
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::{IndexAssignOperator, IndexOperator, Operator};

    use crate::{
        ir::Instruction,
        prelude::{CubeIndex, CubeIndexMut},
    };

    use super::*;

    #[cube]
    impl<E: CubePrimitive> Tensor<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&self, i: u32) -> &E
        where
            Self: CubeIndex,
        {
            intrinsic!(|scope| {
                let out = scope.create_local(self.expand.item);
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: *self.expand,
                        index: i.expand.consume(),
                        line_size: 0,
                    }),
                    *out,
                ));
                out.into()
            })
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_assign_unchecked(&mut self, i: u32, value: E)
        where
            Self: CubeIndexMut,
        {
            intrinsic!(|scope| {
                scope.register(Instruction::new(
                    Operator::UncheckedIndexAssign(IndexAssignOperator {
                        index: i.expand.consume(),
                        value: value.expand.consume(),
                        line_size: 0,
                    }),
                    *self.expand,
                ));
            })
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

impl<T: CubePrimitive> SizedContainer for Tensor<T> {
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
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index_unchecked::expand(scope, self.clone(), idx)
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
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
