//! The [`Transformed`] destination.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_std::tensor::{ErasedTensor, WriteOnly};

use super::transform::{VectorTransform, VectorTransformLaunch};
use super::vectors::transformed_output;
use crate::kind::{GlobalOperand, Write};
use crate::launch::Unbound;
use crate::layout::RuntimeGeometry;
use crate::*;

/// A [`Destination`] over a row-major rank 2 output whose stored vectors each pass through the
/// [`VectorTransform`] `T`, a whole vector at a time, with no change to the routine.
/// `Transformed<()>` stores the vectors unchanged.
pub struct Transformed<T>(PhantomData<fn() -> T>);

/// The output tensor, its geometry, and the spec and vector size the kernel derived it under. The
/// geometry is the tensor's own, checked against the kernel's at launch and passed as scalars.
#[derive(CubeType, CubeLaunch)]
pub struct TransformedOut<'a, E: Numeric> {
    out: &'a Tensor<E>,
    shape: Sequence<u32>,
    strides: Sequence<u32>,
    #[cube(comptime)]
    spec: TileSpec,
    #[cube(comptime)]
    vector_size: usize,
}

/// What a [`Transformed`] launch is made from: the output, its transform's operand, and the
/// sink's [`Unbound`].
pub struct TransformedOperand<T> {
    pub out: TensorArg,
    pub transform: T,
    pub sink: Unbound,
}

#[cube]
impl<'a, E: Numeric> TransformedOut<'a, E> {
    /// The output as vectors written through the transform `T`.
    fn transformed<T: VectorTransform>(&self, transform: &T::Arg<E>) -> ErasedTensor<E, WriteOnly> {
        let size!(W) = comptime!(self.vector_size);
        transformed_output::<E, W, T>(self.out, transform, *self.strides.index(0) as usize)
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
impl<T: VectorTransform> Destination for Transformed<T> {
    type Arg<E: Numeric> = (TransformedOut<'static, E>, T::Arg<E>);
    type Served<E: Numeric> = Tile;

    fn serve<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        let vector_size = V::value();
        let derived = comptime!(arg.0.vector_size);
        comptime!(assert!(
            vector_size == derived,
            "Transformed: the kernel stores vectors of {vector_size}, the output was derived for \
             {derived}"
        ));
        let geometry = arg.0.geometry();
        let sink = arg.0.transformed::<T>(&arg.1);
        GlobalOperand::sink::<E>(
            sink,
            geometry,
            vector_size,
            comptime!(partitioning.space().clone()),
            comptime!(arg.0.spec.clone()),
            Write::Replace,
        )
        .tile::<E>(comptime!(partitioning.levels().to_vec()))
    }
}

impl<T: VectorTransformLaunch> DestinationLaunch for Transformed<T> {
    type Operand = TransformedOperand<T::Operand>;

    fn arg<E: Numeric>(
        operand: Self::Operand,
    ) -> <<Self as Destination>::Arg<E> as LaunchArg>::RuntimeArg {
        (
            TransformedOutLaunch::for_sink(operand.out, operand.sink),
            T::arg::<E>(operand.transform),
        )
    }
}

impl<E: Numeric> TransformedOutLaunch<'static, E> {
    /// `out` as the output a kernel derived `sink` for; panics unless it is rank 2 over the
    /// kernel's shape, its rows contiguous and disjoint, each starting on a vector.
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
                && shape[1].is_multiple_of(vector_size)
                && strides[0].is_multiple_of(vector_size),
            "Transformed: the output must be rank 2 over the kernel's shape {:?}, its rows \
             contiguous, not overlapping, starting on a vector of {vector_size} and a whole \
             number of vectors wide; found shape \
             {shape:?} and strides {strides:?}",
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
