use crate::matmul::cmma_matmul::tile::base::Fragment;
use crate::matmul::matrix::{as_cmma_layout, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};

#[cube]
pub(crate) fn execute<I: Numeric, O: Numeric>(
    lhs: &Fragment<I>,
    rhs: &Fragment<I>,
    out: &mut Fragment<O>,
) {
    cmma::execute::<I, I, O, O>(&lhs.matrix, &rhs.matrix, &out.matrix, &out.matrix);
}

#[cube]
pub(crate) fn init_lhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> Fragment<I> {
    unsafe {
        Fragment::<I> {
            matrix: cmma::Matrix::<I>::uninitialized(
                cmma::MatrixIdent::A,
                m,
                n,
                k,
                as_cmma_layout(layout),
            ),
            stride: match layout {
                MatrixLayout::RowMajor => k,
                MatrixLayout::ColMajor => m,
            },
        }
    }
}

#[cube]
pub(crate) fn init_rhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> Fragment<I> {
    unsafe {
        Fragment::<I> {
            matrix: cmma::Matrix::<I>::uninitialized(
                cmma::MatrixIdent::B,
                m,
                n,
                k,
                as_cmma_layout(layout),
            ),
            stride: match layout {
                MatrixLayout::RowMajor => n,
                MatrixLayout::ColMajor => k,
            },
        }
    }
}

#[cube]
pub(crate) fn fill_lhs<C: CubePrimitive, I: Numeric>(slice: &Slice<'_, C>, lhs: &mut Fragment<I>) {
    cmma::load(&lhs.matrix, slice, lhs.stride);
}

#[cube]
pub(crate) fn fill_rhs<C: CubePrimitive, I: Numeric>(slice: &Slice<'_, C>, rhs: &mut Fragment<I>) {
    cmma::load(&rhs.matrix, slice, rhs.stride);
}

#[cube]
pub(crate) fn init_output<O: Numeric>(m: u32, n: u32, k: u32) -> Fragment<O> {
    unsafe {
        let matrix = cmma::Matrix::<O>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            m,
            n,
            k,
            cmma::MatrixLayout::Undefined,
        );

        cmma::fill(&matrix, O::from_int(0));

        Fragment::<O> { matrix, stride: n }
    }
}

#[cube]
pub(crate) fn read_output<O: Numeric, C: Numeric>(
    out: &Fragment<O>,
    slice: &mut SliceMut<'_, Line<C>>,
) {
    cmma::store(slice, &out.matrix, out.stride, cmma::MatrixLayout::RowMajor);
}
