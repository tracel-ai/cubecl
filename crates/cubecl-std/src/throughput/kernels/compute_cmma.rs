use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch_unchecked)]
pub fn compute_cmma_throughput<F: Float, N: Size>(
    output: &mut [Vector<F, N>],
    #[comptime] n_acc: usize,
    n_iter: usize,
    #[define(F)] _dtype: StorageType,
) {
    let m = 16 as usize;
    let n = 16 as usize;
    let k = 16 as usize;

    let a = cmma::Matrix::<F>::from_value(
        cmma::MatrixIdent::A,
        m,
        n,
        k,
        cmma::MatrixLayout::RowMajor,
        F::new(1.0),
    );

    let b = cmma::Matrix::<F>::from_value(
        cmma::MatrixIdent::B,
        m,
        n,
        k,
        cmma::MatrixLayout::ColMajor,
        F::new(1.0),
    );

    let mut acc = Sequence::<cmma::Matrix<F>>::new();

    #[unroll]
    for _ in 0..n_acc {
        acc.push(cmma::Matrix::<F>::from_value(
            cmma::MatrixIdent::Accumulator,
            m,
            n,
            k,
            cmma::MatrixLayout::Undefined,
            F::new(0.0),
        ));
    }

    for _ in 0..n_iter {
        #[unroll]
        for i in 0..n_acc {
            cmma::execute(&a, &b, acc.index(i), acc.index(i));
        }
    }

    if ABSOLUTE_POS == 0 {
        cmma::store(output, acc.index(0), n as u32, cmma::MatrixLayout::RowMajor);
    }
}
