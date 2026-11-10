//! How the lines a routine stores become the output: [`Merge`].

use cubecl::prelude::*;
use cubecl_std::tensor::{ErasedTensor, WriteOnly};

use crate::launch::Unbound;
use crate::layout::RuntimeGeometry;

/// How the lines a routine stores become the output of a [`Merged`](crate::launch::Merged) destination:
/// written straight out (`()`), or held and combined with other lines once the walk ends. Every line
/// a merge writes goes through `lines`, which applies the stages after the merge, so no stage ever
/// sees a value the merge has yet to combine.
#[cube]
pub trait Merge: Send + Sync + 'static {
    /// The launch argument the merge reads and writes through, at element `E`.
    type Arg<E: Numeric>: LaunchArg + CubeType;

    /// What a unit reads of the merge once, before its walk.
    type Walk: CubeType;

    fn walk<E: Numeric>(arg: &Self::Arg<E>) -> Self::Walk;

    /// Take line `index` of the output, as the routine stored it, in divergent control flow.
    fn line<E: Numeric, N: Size>(
        arg: &Self::Arg<E>,
        walk: &Self::Walk,
        lines: &mut ErasedTensor<E, WriteOnly>,
        index: usize,
        value: Vector<E, N>,
    );

    /// Close the merge, as [`Destination::finish`](crate::launch::Destination::finish) is called:
    /// write what it holds through `lines`. `out` is the output as stored so far, and `geometry`
    /// its shape and strides as launch scalars: a merge never reads them from `out`'s metadata.
    fn finish<E: Numeric>(
        arg: &Self::Arg<E>,
        out: &Tensor<E>,
        geometry: &RuntimeGeometry,
        lines: &mut ErasedTensor<E, WriteOnly>,
    );
}

/// [`Merge`]'s host half: how what the launch is given becomes the argument, for the output the
/// kernel derived.
pub trait MergeLaunch: Merge {
    type Operand;

    fn arg<E: Numeric>(
        operand: Self::Operand,
        out: &Unbound,
    ) -> <Self::Arg<E> as LaunchArg>::RuntimeArg;
}

/// Each line written out as it is stored.
#[cube]
impl Merge for () {
    type Arg<E: Numeric> = ();
    type Walk = ();

    fn walk<E: Numeric>(_arg: &Self::Arg<E>) {}

    fn line<E: Numeric, N: Size>(
        _arg: &Self::Arg<E>,
        _walk: &Self::Walk,
        lines: &mut ErasedTensor<E, WriteOnly>,
        index: usize,
        value: Vector<E, N>,
    ) {
        lines.write::<N>(index, value);
    }

    fn finish<E: Numeric>(
        _arg: &Self::Arg<E>,
        _out: &Tensor<E>,
        _geometry: &RuntimeGeometry,
        _lines: &mut ErasedTensor<E, WriteOnly>,
    ) {
    }
}

impl MergeLaunch for () {
    type Operand = ();

    fn arg<E: Numeric>(_operand: (), _out: &Unbound) {}
}
