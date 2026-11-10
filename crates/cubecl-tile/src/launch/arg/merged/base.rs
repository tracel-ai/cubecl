//! The [`Merged`] destination.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_std::tensor::{ErasedTensor, WriteOnly};

use super::lines::{merge_lines, stage_lines};
use super::merge::{Merge, MergeLaunch};
use super::stage::{LineStage, LineStageLaunch};
use crate::kind::{GlobalOperand, Write};
use crate::launch::Unbound;
use crate::layout::RuntimeGeometry;
use crate::*;

/// A [`Destination`] over a row-major rank 2 output: the routine's lines go to the [`Merge`] `M`,
/// and every line `M` writes out passes through the [`LineStage`] `S`, a whole line at a time.
/// `Merged<(), ()>` stores the lines unchanged.
pub struct Merged<M, S>(PhantomData<fn() -> (M, S)>);

/// The output tensor, its geometry, and the spec and width the kernel derived it under. The
/// geometry is the tensor's own, checked against the kernel's at launch and passed as scalars,
/// since reading it from the tensor's metadata in the kernel is slower.
#[derive(CubeType, CubeLaunch)]
pub struct MergedOut<'a, E: Numeric> {
    out: &'a Tensor<E>,
    shape: Sequence<u32>,
    strides: Sequence<u32>,
    #[cube(comptime)]
    spec: TileSpec,
    #[cube(comptime)]
    width: usize,
}

/// What a [`Merged`] launch is made from: the output, the operands of its merge and of its stages,
/// and the sink's [`Unbound`].
pub struct MergedOperand<M, S> {
    pub out: TensorArg,
    pub merge: M,
    pub stage: S,
    pub sink: Unbound,
}

#[cube]
impl<'a, E: Numeric> MergedOut<'a, E> {
    /// The output as lines written through the stage `S`.
    fn lines<S: LineStage>(&self, stage: &S::Arg<E>) -> ErasedTensor<E, WriteOnly> {
        let size!(W) = comptime!(self.width);
        stage_lines::<E, W, S>(self.out, stage, *self.strides.index(0) as usize)
    }

    /// The output's shape and strides, as the walk addresses it.
    fn geometry(&self) -> RuntimeGeometry {
        let mut geometry = RuntimeGeometry::new();
        #[unroll]
        for axis in 0..2usize {
            geometry.push(*self.shape.index(axis), *self.strides.index(axis));
        }
        geometry
    }
}

#[cube]
impl<M: Merge, S: LineStage> Destination for Merged<M, S> {
    type Arg<E: Numeric> = (MergedOut<'static, E>, M::Arg<E>, S::Arg<E>);

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        let geometry = arg.0.geometry();
        let lines = arg.0.lines::<S>(&arg.2);
        let walk = M::walk::<E>(&arg.1);
        let sink = merge_lines::<E, V, M>(&arg.1, &walk, &lines);
        GlobalOperand::sink::<E>(
            sink,
            geometry,
            V::value(),
            comptime!(partitioning.space().clone()),
            comptime!(arg.0.spec.clone()),
            Write::Replace,
        )
        .tile::<E>(comptime!(partitioning.levels().to_vec()))
    }

    fn finish<E: Numeric>(arg: &Self::Arg<E>) {
        let mut lines = arg.0.lines::<S>(&arg.2);
        M::finish::<E>(&arg.1, arg.0.out, &arg.0.geometry(), &mut lines);
    }
}

impl<M: MergeLaunch, S: LineStageLaunch> DestinationLaunch for Merged<M, S> {
    type Operand = MergedOperand<M::Operand, S::Operand>;

    fn arg<E: Numeric>(
        operand: Self::Operand,
    ) -> <<Self as Destination>::Arg<E> as LaunchArg>::RuntimeArg {
        let merge = M::arg::<E>(operand.merge, &operand.sink);
        (
            MergedOutLaunch::for_sink(operand.out, operand.sink),
            merge,
            S::arg::<E>(operand.stage),
        )
    }
}

impl<E: Numeric> MergedOutLaunch<'static, E> {
    /// `out` as the output a kernel derived `sink` for; panics unless it is rank 2 over the
    /// kernel's shape, its rows contiguous and disjoint, each starting on a line.
    fn for_sink(out: TensorArg, sink: Unbound) -> Self {
        let Unbound {
            spec,
            geometry,
            vector_size,
        } = sink;
        let (shape, strides) = (out.shape().to_vec(), out.strides().to_vec());
        assert!(
            shape == geometry.shape()
                && strides.len() == 2
                && strides[1] == 1
                && strides[0] >= shape[1]
                && strides[0].is_multiple_of(vector_size),
            "Merged: the output must be rank 2 over the kernel's shape {:?}, its rows contiguous, \
             not overlapping, and starting on a line of {vector_size}; found shape {shape:?} and \
             strides {strides:?}",
            geometry.shape(),
        );
        Self::new(
            out,
            shape.iter().map(|&e| e as u32).collect(),
            strides.iter().map(|&s| s as u32).collect(),
            spec,
            vector_size,
        )
    }
}
