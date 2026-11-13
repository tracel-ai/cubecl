//! The write-only output a [`Transformed`](super::Transformed) destination hands out: every vector
//! stored through the transform.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};

use super::transform::{VectorAt, VectorTransform};

/// The output buffer's vectors of `N`, each stored through the transform `T`.
struct TransformedOutput<E: Numeric, N: Size, T: VectorTransform> {
    out: <Tensor<E> as CubeType>::ExpandType,
    transform: <T::Arg<E> as CubeType>::ExpandType,
    row_stride: NativeExpand<usize>,
    _types: PhantomData<fn() -> (N, T)>,
}

impl<E: Numeric, N: Size, T: VectorTransform> TransformedOutput<E, N, T> {
    fn new(
        out: &<Tensor<E> as CubeType>::ExpandType,
        transform: &<T::Arg<E> as CubeType>::ExpandType,
        row_stride: NativeExpand<usize>,
    ) -> Self {
        Self {
            out: ExpandTypeClone::clone_unchecked(out),
            transform: ExpandTypeClone::clone_unchecked(transform),
            row_stride,
            _types: PhantomData,
        }
    }
}

impl<E: Numeric, N: Size, T: VectorTransform> ErasedTensorOperationsExpand<E>
    for TransformedOutput<E, N, T>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    // `__expand_lines_method` and `__expand_write_line_method` are cubecl-std's names for the
    // erased tensor traits. The buffer's vectors, not the tensor's: a pitched buffer holds vectors
    // past its shape's product.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let scalars = self.out.__expand_buffer_len_method(scope);
        let vector_size = <N as Size>::__expand_value(scope).__expand_runtime_method(scope);
        scalars.__expand_div_method(scope, vector_size)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        store_vector::expand::<E, N, T>(
            scope,
            &mut self.out,
            &self.transform,
            self.row_stride,
            index,
            value.into(),
        );
    }
}

impl<E: Numeric, N: Size, T: VectorTransform> WritesLines<E> for TransformedOutput<E, N, T> {}

// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
pub(super) fn transformed_output<E: Numeric, N: Size, T: VectorTransform>(
    _out: &Tensor<E>,
    _transform: &T::Arg<E>,
    _row_stride: usize,
) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

/// The output's vectors, each stored through the transform `T`.
pub(super) mod transformed_output {
    use super::*;

    pub fn expand<E: Numeric, N: Size, T: VectorTransform>(
        _scope: &Scope,
        out: &<Tensor<E> as CubeType>::ExpandType,
        transform: &<T::Arg<E> as CubeType>::ExpandType,
        row_stride: NativeExpand<usize>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(TransformedOutput::<E, N, T>::new(
            out, transform, row_stride,
        ))
    }
}

/// Vector `index` of the output buffer: `N` cells of one row, which the host's vector size choice
/// guarantees by dividing the row, placed by the row stride. Stored a cell at a time, since a
/// scalar buffer cannot be viewed as vectors on every backend.
#[cube]
fn store_vector<E: Numeric, N: Size, T: VectorTransform>(
    out: &mut Tensor<E>,
    transform: &T::Arg<E>,
    row_stride: usize,
    index: usize,
    value: Vector<E, N>,
) {
    let first = index * N::value();
    let at = VectorAt::new(first, row_stride);
    let stored = T::vector::<E, N>(transform, &at, value);
    #[unroll]
    for lane in 0..N::value() {
        out[first + lane] = stored.extract(lane);
    }
}
