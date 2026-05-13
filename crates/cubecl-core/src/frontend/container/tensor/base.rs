use crate::{
    ir::{Metadata, Scope},
    prelude::*,
    unexpanded,
};
use alloc::boxed::Box;
use core::ops::{Deref, DerefMut};
use cubecl_ir::VectorSize;

use crate as cubecl;

/// The tensor type is a wrapper around `[T]` that comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(CubeType)]
pub struct Tensor<T: CubePrimitive> {
    pub(super) meta: TensorMeta,
    pub(super) buffer: [T],
}

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct OwnedTensor<T: CubePrimitive> {
    #[allow(unused)]
    pub(super) meta: TensorMeta,
    pub(super) buffer: Box<[T]>,
}

#[cube]
impl<T: CubePrimitive> OwnedTensor<T> {
    pub fn as_slice(&self) -> &[T] {
        &self.buffer
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buffer
    }
}

#[cube]
impl<T: CubePrimitive> Tensor<T> {
    pub fn as_slice(&self) -> &[T] {
        &self.buffer
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buffer
    }
}

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use cubecl_ir::Variable;

    use super::*;
    use crate::ir::{Arithmetic, BinaryOperands, Instruction};

    #[cube]
    impl<T: CubePrimitive> Tensor<T> {
        /// Obtain the stride of input at dimension dim
        #[allow(unused_variables)]
        pub fn stride(&self, dim: usize) -> usize {
            intrinsic!(|scope| {
                let dim: Variable = dim.into();
                let list = self.__extract_list(scope);
                let out = scope.create_local(usize::__expand_as_type(scope));
                scope.register(Instruction::new(
                    Metadata::Stride {
                        dim: dim,
                        var: list,
                    },
                    out,
                ));
                out.into()
            })
        }

        /// Obtain the shape of input at dimension dim
        #[allow(unused_variables)]
        pub fn shape(&self, dim: usize) -> usize {
            intrinsic!(|scope| {
                let dim: Variable = dim.into();
                let list = self.__extract_list(scope);
                let out = scope.create_local(usize::__expand_as_type(scope));
                scope.register(Instruction::new(
                    Metadata::Shape {
                        dim: dim,
                        var: list,
                    },
                    out,
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
                let index: Variable = index.into();
                let stride = self.__expand_stride_method(scope, dim.clone());
                let shape = self.__expand_shape_method(scope, dim.clone());

                // Compute `num_strides = index / stride`.
                let num_strides = scope.create_local(usize::__expand_as_type(scope));
                scope.register(Instruction::new(
                    Arithmetic::Div(BinaryOperands {
                        lhs: index,
                        rhs: stride.expand.into(),
                    }),
                    num_strides.clone().into(),
                ));

                // Compute `coordinate = num_strides % shape `.
                let coordinate = scope.create_local(usize::__expand_as_type(scope));
                scope.register(Instruction::new(
                    Arithmetic::Rem(BinaryOperands {
                        lhs: num_strides,
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
            self.meta.len
        }

        /// The length of the buffer representing the tensor in terms of vectorized elements.
        ///
        /// # Warning
        ///
        /// The buffer length will be affected by the vectorization factor. To obtain the number of
        /// elements, you should multiply the length by the vectorization factor.
        #[allow(clippy::len_without_is_empty)]
        pub fn buffer_len(&self) -> usize {
            intrinsic!(|scope| { self.__extract_length(scope) })
        }

        /// Returns the rank of the tensor.
        pub fn rank(&self) -> usize {
            self.meta.rank
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
            scope: &Scope,
        ) -> VectorSize {
            expand.__expand_vector_size_method(scope)
        }
    }
}

impl<'a, E: CubePrimitive> From<&'a OwnedTensorExpand<E>> for &'a TensorExpand<E> {
    fn from(value: &'a OwnedTensorExpand<E>) -> Self {
        value.deref()
    }
}

impl<'a, E: CubePrimitive> From<&'a mut OwnedTensorExpand<E>> for &'a mut TensorExpand<E> {
    fn from(value: &'a mut OwnedTensorExpand<E>) -> Self {
        value.deref_mut()
    }
}

impl<T: CubePrimitive> SizedContainer<usize> for Tensor<T> {
    fn len(&self) -> usize {
        unexpanded!()
    }
}

impl<T: CubePrimitive> SizedContainerExpand<usize> for TensorExpand<T> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope)
    }
}

impl<T: CubePrimitive> Iterator for &Tensor<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

impl<T: CubePrimitive> List<T> for Tensor<T> {}

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

impl<T: CubePrimitive> Deref for TensorExpand<T> {
    type Target = SliceExpand<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T: CubePrimitive> DerefMut for TensorExpand<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl<T: CubePrimitive> Deref for OwnedTensor<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> DerefMut for OwnedTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> Deref for OwnedTensorExpand<T> {
    type Target = TensorExpand<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Expand type has compatible layout since the type of the buffer is just a marker
        unsafe { core::mem::transmute(self) }
    }
}

impl<T: CubePrimitive> DerefMut for OwnedTensorExpand<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Expand type has compatible layout since the type of the buffer is just a marker
        unsafe { core::mem::transmute(self) }
    }
}

impl<T: CubePrimitive> ListExpand<T> for TensorExpand<T> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        Self::__expand_len_method(self, scope)
    }
}

impl<T: CubePrimitive> Vectorized for Tensor<T> {}
impl<T: CubePrimitive> VectorizedExpand for TensorExpand<T> {
    fn vector_size(&self) -> VectorSize {
        self.buffer.expand.ty.vector_size()
    }
}
