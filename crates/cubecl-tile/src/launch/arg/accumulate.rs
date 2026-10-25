//! An output several instances accumulate into, as a single launch argument.

use cubecl::define_size;
use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};

use crate::*;

/// An output several instances accumulate into, as a single launch argument, in lines of
/// `width`. The buffer must arrive holding zero: the atomic only ever adds.
#[derive(CubeType, CubeLaunch)]
pub struct AccumulateArg<'a, E: Numeric> {
    pub tensor: &'a Tensor<Atomic<E>>,
    /// The width a line is added in: the buffer's own elements are scalar.
    #[cube(comptime)]
    pub width: usize,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric> AccumulateArg<'a, E> {
    /// Serve the output as a [`Tile`] that accumulates into it in lines of its width.
    pub fn tile(&self, partitioning: &Partitioning) -> Tile {
        let space = comptime!(partitioning.clone());
        // An atomic element is scalar, so these strides are already in scalars.
        let geometry = RuntimeGeometry::of_tensor::<Atomic<E>>(
            self.tensor,
            comptime!(self.spec.projection.physical_rank()),
        );
        let sink = atomic_sink::<E>(self.tensor, comptime!(self.width));
        GlobalOperand::sink(
            sink,
            geometry,
            comptime!(self.width),
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
            Write::Accumulate,
        )
        .tile::<E>(comptime!(space.levels().to_vec()))
    }
}

/// The erased tensor over an atomic buffer, accumulating in lines of `width`.
fn atomic_sink<E: Numeric>(
    _values: &Tensor<Atomic<E>>,
    _width: usize,
) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

mod atomic_sink {
    use super::*;

    pub(super) fn expand<E: Numeric>(
        _scope: &Scope,
        values: &<Tensor<Atomic<E>> as CubeType>::ExpandType,
        width: usize,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(AtomicAccumulate::<E> {
            values: ExpandTypeClone::clone_unchecked(values),
            width,
        })
    }
}

/// A write-only backing that accumulates into an `Atomic<E>` buffer in lines of `width`.
struct AtomicAccumulate<E: Numeric> {
    values: <Tensor<Atomic<E>> as CubeType>::ExpandType,
    width: usize,
}

// The line a write hands over, typed at the width it is added in: registered where the write
// expands and read nowhere after.
define_size!(Line);

impl<E: Numeric> ErasedTensorOperationsExpand<E> for AtomicAccumulate<E> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        self.width
    }

    /// In lines of `width`, as the trait counts them, off a buffer whose own elements are scalar:
    /// the buffer's, not the tensor's, since a pitched buffer holds cells past its shape's product.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let scalars = self.values.__expand_buffer_len_method(scope);
        let width = self.width.__expand_runtime_method(scope);
        scalars.__expand_div_method(scope, width)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        scope.register_size::<Line>(self.width);
        accumulate_line::expand::<E, Line>(scope, &self.values, index, value.into());
    }
}

impl<E: Numeric> WritesLines<E> for AtomicAccumulate<E> {}

/// Accumulate one line into the buffer: `N` scalar adds at the line's own offset.
#[cube]
fn accumulate_line<E: Numeric, N: Size>(
    values: &Tensor<Atomic<E>>,
    index: usize,
    value: Vector<E, N>,
) {
    let base = index * N::value();
    #[unroll]
    for k in 0..N::value() {
        values[base + k].fetch_add(value.extract(k));
    }
}
