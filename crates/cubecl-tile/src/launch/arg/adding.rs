//! A tile whose writes replace each line on a first turn and add into it after: the carry a
//! [`Relay`](crate::launch::Relay)'s turns pass on and a
//! [`LastArrival`](crate::launch::LastArrival)'s slot share it.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};

use crate::kind::Write;
use crate::*;

/// `arg` under `partitioning` as a tile whose writes replace each line where `first` and add into
/// it otherwise: a relay's carry, which its first turn replaces and every later one adds into, and
/// a [`LastArrival`](crate::launch::LastArrival)'s slot, which the last run to arrive adds the
/// others' parts into.
#[cube]
pub(crate) fn adding_tile<E: Numeric, V: Size>(
    arg: &TileArg<'_>,
    partitioning: &Partitioning,
    first: bool,
) -> Tile {
    let space = comptime!(partitioning.clone());
    let lines = arg.lines::<E, V>();
    let geometry = RuntimeGeometry::of_tensor::<Vector<E, V>>(
        lines,
        comptime!(arg.spec.projection.physical_rank()),
    );
    GlobalOperand::sink(
        relay_sink::<E, V>(lines, first),
        geometry,
        V::value(),
        comptime!(space.space().clone()),
        comptime!(arg.spec.clone()),
        comptime!(Write::Exclusive(Schedule::Sequential)),
    )
    .tile::<E>(comptime!(space.levels().to_vec()))
}

/// The erased tensor over the carry that replaces each line written on the `first` turn and adds
/// into it on every other.
///
/// A constructor here rather than in cubecl for the reason the atomic sink's is: what a write
/// *means* is this crate's statement, and cubecl's own backings all replace.
// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
fn relay_sink<E: Numeric, N: Size>(
    _values: &Tensor<Vector<E, N>>,
    _first: bool,
) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

mod relay_sink {
    use super::*;

    pub fn expand<E: Numeric, N: Size>(
        _scope: &Scope,
        values: &<Tensor<Vector<E, N>> as CubeType>::ExpandType,
        first: NativeExpand<bool>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(RelaySink::<E, N> {
            values: ExpandTypeClone::clone_unchecked(values),
            first,
            _n: PhantomData,
        })
    }
}

/// A backing that writes each line on the first turn and adds into it on the rest: a read and a
/// write, not an atomic, so it declares [`WritesLines`] alone. The read is the write's own, never
/// the drain's, and the turns keep one cube at the box at a time.
struct RelaySink<E: Numeric, N: Size> {
    values: <Tensor<Vector<E, N>> as CubeType>::ExpandType,
    first: NativeExpand<bool>,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size> ErasedTensorOperationsExpand<E> for RelaySink<E, N> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    /// The buffer's lines, not the tensor's: a pitched buffer holds lines past its shape's
    /// product, and the rows a layout addresses there are written like any other.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.values.__expand_buffer_len_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        relay_line::expand::<E, N>(scope, &mut self.values, index, value.into(), self.first);
    }
}

impl<E: Numeric, N: Size> WritesLines<E> for RelaySink<E, N> {}

/// One line of the carry: `value` on the first turn, the line there plus `value` after.
#[cube]
fn relay_line<E: Numeric, N: Size>(
    values: &mut Tensor<Vector<E, N>>,
    index: usize,
    value: Vector<E, N>,
    first: bool,
) {
    if first {
        values[index] = value;
    } else {
        values[index] += value;
    }
}
