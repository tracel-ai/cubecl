//! The merge that reduces each row to one value: [`ReducedRows`].

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_std::tensor::{ErasedTensor, WriteOnly};

use super::base::{Merge, MergeLaunch};
use super::row_reduction::RowReduction;
use super::row_reduction_scratch::{RowReductionScratch, RowReductionScratchArg};
use crate::launch::Unbound;
use crate::layout::RuntimeGeometry;
use crate::*;

/// A [`Merge`] that writes each vector out as it is stored and, once the walk ends, writes each
/// row's [`RowReduction`] `F` of what it wrote into [`RowReductionScratch`]. The reduction sees the values the
/// merge writes out, before any stage transforms them. The last holder of a row of boxes to arrive
/// sums the parked parts, lanes striding along them and then a plane tree, so the value has the
/// same bits whichever holder that is, though it depends on the plane width.
pub struct ReducedRows<F>(PhantomData<fn() -> F>);

#[cube]
impl<F: RowReduction> Merge for ReducedRows<F> {
    type Arg<E: Numeric> = (RowReductionScratchArg, F::Param);
    type Walk = ();

    fn walk<E: Numeric>(_arg: &Self::Arg<E>) {}

    fn vector<E: Numeric, N: Size>(
        _arg: &Self::Arg<E>,
        _walk: &Self::Walk,
        out: &mut ErasedTensor<E, WriteOnly>,
        index: usize,
        value: Vector<E, N>,
    ) {
        out.write::<N>(index, value);
    }

    fn finish<E: Numeric>(
        arg: &Self::Arg<E>,
        stored: &Tensor<E>,
        geometry: &RuntimeGeometry,
        _out: &mut ErasedTensor<E, WriteOnly>,
    ) {
        arg.0.merge::<E, F>(stored, geometry, &arg.1);
    }
}

impl<F: RowReduction> MergeLaunch for ReducedRows<F> {
    type Operand = (RowReductionScratch, <F::Param as LaunchArg>::RuntimeArg);

    fn arg<E: Numeric>(
        (buffers, param): Self::Operand,
        out: &Unbound,
    ) -> <Self::Arg<E> as LaunchArg>::RuntimeArg {
        (buffers.arg_for(out), param)
    }
}
