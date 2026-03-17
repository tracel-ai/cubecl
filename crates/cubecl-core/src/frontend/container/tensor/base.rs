use crate::{
    frontend::{CubePrimitive, CubeType, NativeExpand, SizedContainer},
    ir::{Metadata, Scope},
    prelude::*,
    unexpanded,
};
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};
use cubecl_ir::VectorSize;
use cubecl_macros::{cube, intrinsic};

use crate as cubecl;

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new, Clone, Copy)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

type TensorExpand<T> = NativeExpand<Tensor<T>>;

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use cubecl_ir::ManagedVariable;

    use super::*;
    use crate::{
        ir::{Arithmetic, BinaryOperator, Instruction},
        prelude::Array,
    };

    #[cube]
    impl<T: CubeType> Tensor<T> {
        /// Obtain the stride of input at dimension dim
        #[allow(unused_variables)]
        pub fn stride(&self, dim: usize) -> usize {
            intrinsic!(|scope| {
                let dim: ManagedVariable = dim.into();
                let out = scope.create_local(usize::as_type(scope));
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
        pub fn shape(&self, dim: usize) -> usize {
            intrinsic!(|scope| {
                let dim: ManagedVariable = dim.into();
                let out = scope.create_local(usize::as_type(scope));
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
        pub fn coordinate(&self, index: usize, dim: usize) -> usize {
            intrinsic!(|scope| {
                let index: ManagedVariable = index.into();
                let stride = self.clone().__expand_stride_method(scope, dim.clone());
                let shape = self.clone().__expand_shape_method(scope, dim.clone());

                // Compute `num_strides = index / stride`.
                let num_strides = scope.create_local(usize::as_type(scope));
                scope.register(Instruction::new(
                    Arithmetic::Div(BinaryOperator {
                        lhs: *index,
                        rhs: stride.expand.into(),
                    }),
                    num_strides.clone().into(),
                ));

                // Compute `coordinate = num_strides % shape `.
                let coordinate = scope.create_local(usize::as_type(scope));
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
        pub fn len(&self) -> usize {
            intrinsic!(|scope| {
                let elem: NativeExpand<Array<u32>> = self.expand.into();
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
        pub fn buffer_len(&self) -> usize {
            intrinsic!(|scope| {
                let elem: NativeExpand<Array<u32>> = self.expand.into();
                elem.__expand_buffer_len_method(scope)
            })
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> usize {
            intrinsic!(|scope| {
                let out = scope.create_local(usize::as_type(scope));
                scope.register(Instruction::new(Metadata::Rank { var: *self.expand }, *out));
                out.into()
            })
        }
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::{IndexAssignOperator, IndexOperator, Operator};

    use crate::ir::Instruction;

    use super::*;

    #[cube]
    impl<E: CubePrimitive> Tensor<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&self, i: usize) -> &E {
            intrinsic!(|scope| {
                let out = scope.create_local(self.expand.ty);
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: *self.expand,
                        index: i.expand.consume(),
                        vector_size: 0,
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
        pub unsafe fn index_assign_unchecked(&mut self, i: usize, value: E) {
            intrinsic!(|scope| {
                scope.register(Instruction::new(
                    Operator::UncheckedIndexAssign(IndexAssignOperator {
                        index: i.expand.consume(),
                        value: value.expand.consume(),
                        vector_size: 0,
                        unroll_factor: 1,
                    }),
                    *self.expand,
                ));
            })
        }
    }
}

/// Module that contains the implementation details of the `vector_size` function.
mod vector {
    use super::*;

    impl<P: Scalar, N: Size> Tensor<Vector<P, N>> {
        /// Get the size of each vector contained in the tensor.
        ///
        /// Same as the following:
        ///
        /// ```rust, ignore
        /// let size = tensor[0].size();
        /// ```
        pub fn vector_size(&self) -> VectorSize {
            N::value()
        }

        // Expand function of [size](Tensor::vector_size).
        pub fn __expand_vector_size(
            expand: <Self as CubeType>::ExpandType,
            scope: &mut Scope,
        ) -> VectorSize {
            expand.__expand_vector_size_method(scope)
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
    type ExpandType = NativeExpand<Tensor<T>>;
}

impl<T: CubeType> CubeType for *const Tensor<T> {
    type ExpandType = NativeExpand<Tensor<T>>;
}

impl<T: CubeType> CubeType for *mut Tensor<T> {
    type ExpandType = NativeExpand<Tensor<T>>;
}

impl<T: CubeType> CubeType for &mut Tensor<T> {
    type ExpandType = NativeExpand<Tensor<T>>;
}

impl<T: CubeType> CubeType for &Tensor<T> {
    type ExpandType = NativeExpand<Tensor<T>>;
}

impl<C: CubeType> IntoMut for NativeExpand<Tensor<C>> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> List<T> for Tensor<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: NativeExpand<Tensor<T>>,
        idx: NativeExpand<usize>,
    ) -> NativeExpand<T> {
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

impl<T: CubePrimitive> ListExpand<T> for NativeExpand<Tensor<T>> {
    fn __expand_read_method(&self, scope: &mut Scope, idx: NativeExpand<usize>) -> NativeExpand<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> NativeExpand<T> {
        index_unchecked::expand(scope, self.clone(), idx)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> NativeExpand<usize> {
        Self::__expand_len(scope, self.clone())
    }
}

impl<T: CubePrimitive> Vectorized for Tensor<T> {}
impl<T: CubePrimitive> VectorizedExpand for NativeExpand<Tensor<T>> {
    fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }
}

impl<T: CubePrimitive> ListMut<T> for Tensor<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: NativeExpand<Tensor<T>>,
        idx: NativeExpand<usize>,
        value: NativeExpand<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for NativeExpand<Tensor<T>> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
        value: NativeExpand<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
