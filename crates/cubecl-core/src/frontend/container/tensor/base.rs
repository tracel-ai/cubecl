use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, SizedContainer},
    ir::{Metadata, Scope, Type},
    prelude::{
        Line, Lined, LinedExpand, List, ListExpand, ListMut, ListMutExpand, index, index_assign,
        index_unchecked,
    },
    unexpanded,
};
use cubecl_ir::{ExpandElement, LineSize};
use cubecl_macros::{cube, intrinsic};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate as cubecl;

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new, Clone, Copy)]
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
        /// Obtain the stride of input at the given axis
        #[allow(unused_variables)]
        pub fn stride(&self, axis: usize) -> usize {
            intrinsic!(|scope| {
                let axis: ExpandElement = axis.into();
                let out = scope.create_local(Type::new(usize::as_type(scope)));
                scope.register(Instruction::new(
                    Metadata::Stride {
                        axis: *axis,
                        var: self.expand.into(),
                    },
                    out.clone().into(),
                ));
                out.into()
            })
        }

        /// Obtain the shape of input at the given axis
        #[allow(unused_variables)]
        pub fn shape(&self, axis: usize) -> usize {
            intrinsic!(|scope| {
                let axis: ExpandElement = axis.into();
                let out = scope.create_local(Type::new(usize::as_type(scope)));
                scope.register(Instruction::new(
                    Metadata::Shape {
                        axis: *axis,
                        var: self.expand.into(),
                    },
                    out.clone().into(),
                ));
                out.into()
            })
        }

        /// Obtain the coordinate corresponding to the given `index` of the tensor at the given `axis`.
        ///
        /// A coordinate is a list of indices corresponding to the multi-dimensional position of an element in the tensor.
        /// The `axis` element in a coordinate is the position along that axis of the tensor.
        #[allow(unused_variables)]
        pub fn coordinate(&self, index: usize, axis: usize) -> usize {
            intrinsic!(|scope| {
                let index: ExpandElement = index.into();
                let stride = self.clone().__expand_stride_method(scope, axis.clone());
                let shape = self.clone().__expand_shape_method(scope, axis.clone());

                // Compute `stride_count = index / stride`.
                let stride_count = scope.create_local(Type::new(usize::as_type(scope)));
                scope.register(Instruction::new(
                    Arithmetic::Div(BinaryOperator {
                        lhs: *index,
                        rhs: stride.expand.into(),
                    }),
                    stride_count.clone().into(),
                ));

                // Compute `coordinate = stride_count % shape `.
                let coordinate = scope.create_local(Type::new(usize::as_type(scope)));
                scope.register(Instruction::new(
                    Arithmetic::Modulo(BinaryOperator {
                        lhs: *stride_count,
                        rhs: shape.expand.into(),
                    }),
                    coordinate.clone().into(),
                ));

                coordinate.into()
            })
        }

        /// The number of lined elements in the tensor.
        ///
        /// # Warning
        ///
        /// The length will be affected by the line size. To obtain the number of elements,
        /// you should multiply the length by the line size.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> usize {
            intrinsic!(|scope| {
                let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
                elem.__expand_len_method(scope)
            })
        }

        /// The length of the buffer representing the tensor in terms of lined elements.
        ///
        /// # Warning
        ///
        /// The buffer length will be affected by the line size. To obtain the number of
        /// elements, you should multiply the length by the line size.
        #[allow(clippy::len_without_is_empty)]
        pub fn buffer_len(&self) -> usize {
            intrinsic!(|scope| {
                let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
                elem.__expand_buffer_len_method(scope)
            })
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> usize {
            intrinsic!(|scope| {
                let out = scope.create_local(Type::new(usize::as_type(scope)));
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
        pub unsafe fn index_unchecked(&self, i: usize) -> &E
        where
            Self: CubeIndex,
        {
            intrinsic!(|scope| {
                let out = scope.create_local(self.expand.ty);
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: *self.expand,
                        index: i.expand.consume(),
                        line_size: 0,
                        unroll_factor: 1,
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
        pub unsafe fn index_assign_unchecked(&mut self, i: usize, value: E)
        where
            Self: CubeIndexMut,
        {
            intrinsic!(|scope| {
                scope.register(Instruction::new(
                    Operator::UncheckedIndexAssign(IndexAssignOperator {
                        index: i.expand.consume(),
                        value: value.expand.consume(),
                        line_size: 0,
                        unroll_factor: 1,
                    }),
                    *self.expand,
                ));
            })
        }
    }
}

/// Module that contains the implementation details of the `line_size` function.
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
        pub fn line_size(&self) -> LineSize {
            unexpanded!()
        }

        // Expand function of [size](Tensor::line_size).
        pub fn __expand_line_size(
            expand: <Self as CubeType>::ExpandType,
            scope: &mut Scope,
        ) -> LineSize {
            expand.__expand_line_size_method(scope)
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

impl<T: CubeType> CubeType for &mut Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<T: CubeType> CubeType for &Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<C: CubeType> ExpandElementIntoMut for Tensor<C> {
    fn elem_into_mut(_scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        elem
    }
}

impl<T: CubePrimitive> List<T> for Tensor<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: ExpandElementTyped<Tensor<T>>,
        idx: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, this, idx)
    }
}

impl<T: CubePrimitive> Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> ListExpand<T> for ExpandElementTyped<Tensor<T>> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<T> {
        index_unchecked::expand(scope, self.clone(), idx)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        Self::__expand_len(scope, self.clone())
    }
}

impl<T: CubePrimitive> Lined for Tensor<T> {}
impl<T: CubePrimitive> LinedExpand for ExpandElementTyped<Tensor<T>> {
    fn line_size(&self) -> LineSize {
        self.expand.ty.line_size()
    }
}

impl<T: CubePrimitive> ListMut<T> for Tensor<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: ExpandElementTyped<Tensor<T>>,
        idx: ExpandElementTyped<usize>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for ExpandElementTyped<Tensor<T>> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<usize>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
