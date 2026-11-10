//! The write-only outputs a [`Merged`](super::Merged) destination hands out: one stores through the
//! stage, the other hands each vector to the merge.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};

use super::merge::Merge;
use super::stage::{VectorAt, VectorStage};

/// The output buffer's `N` wide vectors, each stored through the stage `S`.
struct StagedOutput<E: Numeric, N: Size, S: VectorStage> {
    out: <Tensor<E> as CubeType>::ExpandType,
    stage: <S::Arg<E> as CubeType>::ExpandType,
    row_stride: NativeExpand<usize>,
    _vector_size: PhantomData<fn() -> (N, S)>,
}

impl<E: Numeric, N: Size, S: VectorStage> StagedOutput<E, N, S> {
    fn new(
        out: &<Tensor<E> as CubeType>::ExpandType,
        stage: &<S::Arg<E> as CubeType>::ExpandType,
        row_stride: NativeExpand<usize>,
    ) -> Self {
        Self {
            out: ExpandTypeClone::clone_unchecked(out),
            stage: ExpandTypeClone::clone_unchecked(stage),
            row_stride,
            _vector_size: PhantomData,
        }
    }
}

impl<E: Numeric, N: Size, S: VectorStage> ErasedTensorOperationsExpand<E>
    for StagedOutput<E, N, S>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    // `__expand_lines_method` and `__expand_write_line_method` are cubecl-std's names for the erased
    // tensor traits. The buffer's vectors, not the tensor's: a pitched buffer holds vectors past
    // its shape's product.
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
        store_vector::expand::<E, N, S>(
            scope,
            &mut self.out,
            &self.stage,
            self.row_stride,
            index,
            value.into(),
        );
    }
}

impl<E: Numeric, N: Size, S: VectorStage> WritesLines<E> for StagedOutput<E, N, S> {}

/// Each vector handed to the merge `M`, which writes out through `out`.
struct MergeInput<E: Numeric, N: Size, M: Merge> {
    merge: <M::Arg<E> as CubeType>::ExpandType,
    walk: <M::Walk as CubeType>::ExpandType,
    out: ErasedTensorExpand<E, WriteOnly>,
    _vector_size: PhantomData<fn() -> (N, M)>,
}

impl<E: Numeric, N: Size, M: Merge> MergeInput<E, N, M> {
    fn new(
        merge: &<M::Arg<E> as CubeType>::ExpandType,
        walk: &<M::Walk as CubeType>::ExpandType,
        out: &ErasedTensorExpand<E, WriteOnly>,
    ) -> Self {
        Self {
            merge: ExpandTypeClone::clone_unchecked(merge),
            walk: ExpandTypeClone::clone_unchecked(walk),
            out: out.clone(),
            _vector_size: PhantomData,
        }
    }
}

impl<E: Numeric, N: Size, M: Merge> ErasedTensorOperationsExpand<E> for MergeInput<E, N, M> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.out.__expand_len_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        M::__expand_vector::<E, N>(
            scope,
            &self.merge,
            &self.walk,
            &mut self.out,
            index,
            value.into(),
        );
    }
}

impl<E: Numeric, N: Size, M: Merge> WritesLines<E> for MergeInput<E, N, M> {}

// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
pub(super) fn staged_output<E: Numeric, N: Size, S: VectorStage>(
    _out: &Tensor<E>,
    _stage: &S::Arg<E>,
    _row_stride: usize,
) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

/// The output's vectors, each stored through the stage `S`.
pub(super) mod staged_output {
    use super::*;

    pub fn expand<E: Numeric, N: Size, S: VectorStage>(
        _scope: &Scope,
        out: &<Tensor<E> as CubeType>::ExpandType,
        stage: &<S::Arg<E> as CubeType>::ExpandType,
        row_stride: NativeExpand<usize>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(StagedOutput::<E, N, S>::new(out, stage, row_stride))
    }
}

/// Vector `index` of the output buffer: `N` cells of one row, which the host's vector size choice
/// guarantees by dividing the row, placed by the row stride. Stored a cell at a time, since a
/// scalar buffer cannot be viewed as vectors on every backend.
#[cube]
fn store_vector<E: Numeric, N: Size, S: VectorStage>(
    out: &mut Tensor<E>,
    stage: &S::Arg<E>,
    row_stride: usize,
    index: usize,
    value: Vector<E, N>,
) {
    let first = index * N::value();
    let at = VectorAt::new(first, row_stride);
    let stored = S::vector::<E, N>(stage, &at, value);
    #[unroll]
    for lane in 0..N::value() {
        out[first + lane] = stored.extract(lane);
    }
}

// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
pub(super) fn merge_input<E: Numeric, N: Size, M: Merge>(
    _merge: &M::Arg<E>,
    _walk: &M::Walk,
    _out: &ErasedTensor<E, WriteOnly>,
) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

/// Vectors handed to the merge `M`, which writes out through `out`.
pub(super) mod merge_input {
    use super::*;

    pub fn expand<E: Numeric, N: Size, M: Merge>(
        _scope: &Scope,
        merge: &<M::Arg<E> as CubeType>::ExpandType,
        walk: &<M::Walk as CubeType>::ExpandType,
        out: &ErasedTensorExpand<E, WriteOnly>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(MergeInput::<E, N, M>::new(merge, walk, out))
    }
}
