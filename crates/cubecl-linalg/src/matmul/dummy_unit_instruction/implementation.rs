use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::dummy_unit_instruction::DummyMatrix;
use crate::matmul::matrix_layout::MatrixLayout;

#[cube]
pub(crate) fn execute<I: Numeric, O: Numeric>(
    lhs: &DummyMatrix<I>,
    rhs: &DummyMatrix<I>,
    out: &mut DummyMatrix<O>,
) {
    let m = lhs.shape.0;
    let n = rhs.shape.1;
    let k = rhs.shape.0;

    for i in 0..m {
        for j in 0..n {
            for k_ in 0..k {
                out.handle[i * n + j] +=
                    O::cast_from(lhs.handle[i * k + k_] * rhs.handle[k_ * n + j]);
            }
        }
    }
}

#[cube]
pub(crate) fn init_lhs<I: Numeric>(
    #[comptime] _layout: MatrixLayout,
    m: u32,
    _n: u32,
    k: u32,
) -> DummyMatrix<I> {
    DummyMatrix::<I> {
        handle: Array::<I>::new(256),
        shape: (m, k),
    }
}

#[cube]
pub(crate) fn init_rhs<I: Numeric>(
    #[comptime] _layout: MatrixLayout,
    _m: u32,
    n: u32,
    k: u32,
) -> DummyMatrix<I> {
    DummyMatrix::<I> {
        handle: Array::<I>::new(256),
        shape: (k, n),
    }
}

#[cube]
pub(crate) fn fill_lhs<C: CubePrimitive, I: Numeric>(
    slice: &Slice<'_, C>,
    lhs: &mut DummyMatrix<I>,
    _k: u32,
) {
    for i in 0..256 {
        lhs.handle[i] = I::cast_from(slice[i]);
    }
}

#[cube]
pub(crate) fn fill_rhs<C: CubePrimitive, I: Numeric>(
    slice: &Slice<'_, C>,
    rhs: &mut DummyMatrix<I>,
    _n: u32,
) {
    for i in 0..256 {
        rhs.handle[i] = I::cast_from(slice[i]);
    }
}

#[cube]
pub(crate) fn init_output<O: Numeric>(m: u32, n: u32, _k: u32) -> DummyMatrix<O> {
    DummyMatrix::<O> {
        handle: Array::<O>::new(256),
        shape: (m, n),
    }
}

#[cube]
pub(crate) fn read_output<O: Numeric, C: CubePrimitive>(
    out: &DummyMatrix<O>,
    slice: &mut SliceMut<'_, C>,
    _n: u32,
) {
    for i in 0..256 {
        slice[i] = C::cast_from(out.handle[i]);
    }
}
