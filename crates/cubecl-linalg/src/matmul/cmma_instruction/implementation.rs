use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};

use crate::matmul::matrix_layout::{as_cmma_layout, MatrixLayout};

#[cube]
pub(super) fn execute<I: Numeric, O: Numeric>(
    lhs: &cmma::Matrix<I>,
    rhs: &cmma::Matrix<I>,
    out: &mut cmma::Matrix<O>,
) {
    cmma::execute::<I, I, O, O>(lhs, rhs, out, out);
}

#[cube]
pub(super) fn init_lhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> cmma::Matrix<I> {
    cmma::Matrix::<I>::new(cmma::MatrixIdent::A, m, n, k, as_cmma_layout(layout))
}

#[cube]
pub(super) fn init_rhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> cmma::Matrix<I> {
    cmma::Matrix::<I>::new(cmma::MatrixIdent::B, m, n, k, as_cmma_layout(layout))
}

#[cube]
pub(super) fn fill_lhs<C: CubePrimitive, I: Numeric>(
    slice: &Slice<'_, C>,
    lhs: &mut cmma::Matrix<I>,
    m: u32,
    k: u32,
) {
    match layout {
       MatrixLayout::RowMajor =>cmma::load(&lhs, slice, k),
        MatrixLayout::ColMajor =>cmma::load(&lhs, slice, m),
    }
}

#[cube]
pub(super) fn fill_rhs<C: CubePrimitive, I: Numeric>(
    slice: &Slice<'_, C>,
    rhs: &mut cmma::Matrix<I>,
    k: u32,
    n: u32,
) {
    match layout {
       MatrixLayout::RowMajor =>cmma::load(&rhs, slice, n),
        MatrixLayout::ColMajor =>cmma::load(&rhs, slice, k),
    }
}

#[cube]
pub(super) fn init_output<O: Numeric>(m: u32, n: u32, k: u32) -> cmma::Matrix<O> {
    let out = cmma::Matrix::<O>::new(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
    );

    cmma::fill(&out, O::from_int(0));

    out
}

#[cube]
pub(super) fn read_output<O: Numeric, C: CubePrimitive>(
    out: &cmma::Matrix<O>,
    slice: &mut SliceMut<'_, C>,
    n: u32,
) {
    cmma::store(slice, out, n, cmma::MatrixLayout::RowMajor);
}
