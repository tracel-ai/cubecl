//! The write-only outputs a [`Merged`](super::Merged) destination hands out: one stores through the
//! stage, the other hands each line to the merge.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};

use super::merge::Merge;
use super::stage::{LineAt, LineStage};

/// The output buffer's `N` wide lines, each stored through the stage `S`.
struct StageStore<E: Numeric, N: Size, S: LineStage> {
    out: <Tensor<E> as CubeType>::ExpandType,
    stage: <S::Arg<E> as CubeType>::ExpandType,
    row_stride: NativeExpand<usize>,
    _width: PhantomData<fn() -> (N, S)>,
}

impl<E: Numeric, N: Size, S: LineStage> StageStore<E, N, S> {
    fn new(
        out: &<Tensor<E> as CubeType>::ExpandType,
        stage: &<S::Arg<E> as CubeType>::ExpandType,
        row_stride: NativeExpand<usize>,
    ) -> Self {
        Self {
            out: ExpandTypeClone::clone_unchecked(out),
            stage: ExpandTypeClone::clone_unchecked(stage),
            row_stride,
            _width: PhantomData,
        }
    }
}

impl<E: Numeric, N: Size, S: LineStage> ErasedTensorOperationsExpand<E> for StageStore<E, N, S> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    // The buffer's lines, not the tensor's: a pitched buffer holds lines past its shape's product.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let scalars = self.out.__expand_buffer_len_method(scope);
        let width = <N as Size>::__expand_value(scope).__expand_runtime_method(scope);
        scalars.__expand_div_method(scope, width)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        store_line::expand::<E, N, S>(
            scope,
            &mut self.out,
            &self.stage,
            self.row_stride,
            index,
            value.into(),
        );
    }
}

impl<E: Numeric, N: Size, S: LineStage> WritesLines<E> for StageStore<E, N, S> {}

/// Each line handed to the merge `M`, which writes out through `lines`.
struct MergeFeed<E: Numeric, N: Size, M: Merge> {
    merge: <M::Arg<E> as CubeType>::ExpandType,
    walk: <M::Walk as CubeType>::ExpandType,
    lines: ErasedTensorExpand<E, WriteOnly>,
    _width: PhantomData<fn() -> (N, M)>,
}

impl<E: Numeric, N: Size, M: Merge> MergeFeed<E, N, M> {
    fn new(
        merge: &<M::Arg<E> as CubeType>::ExpandType,
        walk: &<M::Walk as CubeType>::ExpandType,
        lines: &ErasedTensorExpand<E, WriteOnly>,
    ) -> Self {
        Self {
            merge: ExpandTypeClone::clone_unchecked(merge),
            walk: ExpandTypeClone::clone_unchecked(walk),
            lines: lines.clone(),
            _width: PhantomData,
        }
    }
}

impl<E: Numeric, N: Size, M: Merge> ErasedTensorOperationsExpand<E> for MergeFeed<E, N, M> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.lines.__expand_len_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        M::__expand_line::<E, N>(
            scope,
            &self.merge,
            &self.walk,
            &mut self.lines,
            index,
            value.into(),
        );
    }
}

impl<E: Numeric, N: Size, M: Merge> WritesLines<E> for MergeFeed<E, N, M> {}

// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
pub(super) fn stage_lines<E: Numeric, N: Size, S: LineStage>(
    _out: &Tensor<E>,
    _stage: &S::Arg<E>,
    _row_stride: usize,
) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

/// The output's lines, each stored through the stage `S`.
pub(super) mod stage_lines {
    use super::*;

    pub fn expand<E: Numeric, N: Size, S: LineStage>(
        _scope: &Scope,
        out: &<Tensor<E> as CubeType>::ExpandType,
        stage: &<S::Arg<E> as CubeType>::ExpandType,
        row_stride: NativeExpand<usize>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(StageStore::<E, N, S>::new(out, stage, row_stride))
    }
}

/// Line `index` of the output buffer: `N` cells of one row, which the host's width choice
/// guarantees by dividing the row, placed by the row stride. Stored a cell at a time, since a
/// scalar buffer cannot be viewed as vectors on every backend.
#[cube]
fn store_line<E: Numeric, N: Size, S: LineStage>(
    out: &mut Tensor<E>,
    stage: &S::Arg<E>,
    row_stride: usize,
    index: usize,
    value: Vector<E, N>,
) {
    let first = index * N::value();
    let at = LineAt::new(first, row_stride);
    let stored = S::line::<E, N>(stage, &at, value);
    #[unroll]
    for lane in 0..N::value() {
        out[first + lane] = stored.extract(lane);
    }
}

// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
pub(super) fn merge_lines<E: Numeric, N: Size, M: Merge>(
    _merge: &M::Arg<E>,
    _walk: &M::Walk,
    _lines: &ErasedTensor<E, WriteOnly>,
) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

/// Lines handed to the merge `M`, which writes out through `lines`.
pub(super) mod merge_lines {
    use super::*;

    pub fn expand<E: Numeric, N: Size, M: Merge>(
        _scope: &Scope,
        merge: &<M::Arg<E> as CubeType>::ExpandType,
        walk: &<M::Walk as CubeType>::ExpandType,
        lines: &ErasedTensorExpand<E, WriteOnly>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(MergeFeed::<E, N, M>::new(merge, walk, lines))
    }
}
