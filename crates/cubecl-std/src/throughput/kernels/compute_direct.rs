use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch_unchecked)]
pub fn compute_direct_throughput<F: Float, N: Size>(
    output: &mut [Vector<F, N>],
    #[comptime] n_acc: usize,
    n_iter: usize,
    #[define(F)] _dtype: StorageType,
) {
    let tid = F::cast_from(ABSOLUTE_POS);

    let b = Vector::new(tid * F::new(1e-7) + F::new(1.0));
    let c = Vector::new(tid * F::new(1e-7));

    let mut acc = Array::new(n_acc);
    #[unroll]
    for i in 0..n_acc {
        acc[i] = Vector::new(tid + F::cast_from(i));
    }

    for _ in 0..n_iter {
        #[unroll]
        for i in 0..n_acc {
            acc[i] = acc[i] * b + c;
        }
    }

    let mut sum = Vector::zeroed();
    #[unroll]
    for i in 0..n_acc {
        sum += acc[i];
    }

    if ABSOLUTE_POS == 0 {
        output[0] = sum;
    }
}
