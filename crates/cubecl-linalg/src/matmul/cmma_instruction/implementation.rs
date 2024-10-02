use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};

use crate::matmul::cmma_instruction::base::Fragment;
use crate::matmul::matrix_layout::{as_cmma_layout, MatrixLayout};

#[cube]
pub(super) fn execute<I: Numeric, O: Numeric>(
    lhs: &Fragment<I>,
    rhs: &Fragment<I>,
    out: &mut Fragment<O>,
) {
    cmma::execute::<I, I, O, O>(&lhs.matrix, &rhs.matrix, &out.matrix, &out.matrix);
}

#[cube]
pub(super) fn init_lhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> Fragment<I> {
    Fragment::<I> {
        matrix: cmma::Matrix::<I>::new(cmma::MatrixIdent::A, m, n, k, as_cmma_layout(layout)),
        stride: match layout {
            MatrixLayout::RowMajor => k,
            MatrixLayout::ColMajor => m,
        },
    }
}

#[cube]
pub(super) fn init_rhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> Fragment<I> {
    Fragment::<I> {
        matrix: cmma::Matrix::<I>::new(cmma::MatrixIdent::B, m, n, k, as_cmma_layout(layout)),
        stride: match layout {
            MatrixLayout::RowMajor => n,
            MatrixLayout::ColMajor => k,
        },
    }
}

#[cube]
pub(super) fn fill_lhs<C: CubePrimitive, I: Numeric>(slice: &Slice<'_, C>, lhs: &mut Fragment<I>) {
    cmma::load(&lhs.matrix, slice, lhs.stride);
}

#[cube]
pub(super) fn fill_rhs<C: CubePrimitive, I: Numeric>(slice: &Slice<'_, C>, rhs: &mut Fragment<I>) {
    cmma::load(&rhs.matrix, slice, rhs.stride);
}

#[cube]
pub(super) fn init_output<O: Numeric>(m: u32, n: u32, k: u32) -> Fragment<O> {
    let matrix = cmma::Matrix::<O>::new(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
    );

    cmma::fill(&matrix, O::from_int(0));

    Fragment::<O> { matrix, stride: n }
}

#[cube]
pub(super) fn read_output<O: Numeric, C: CubePrimitive>(
    out: &Fragment<O>,
    slice: &mut SliceMut<'_, C>,
) {
    cmma::store(slice, &out.matrix, out.stride, cmma::MatrixLayout::RowMajor);
}
