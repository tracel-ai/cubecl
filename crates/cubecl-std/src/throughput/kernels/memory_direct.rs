use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch_unchecked)]
pub fn memory_direct_throughput<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    n_iter: usize,
    #[define(F)] _dtype: StorageType,
) {
    let len = output.len();
    let stride = CUBE_DIM as usize * CUBE_COUNT;

    for _ in 0..n_iter {
        let mut idx = ABSOLUTE_POS;
        while idx < len {
            output[idx] = input[idx];
            idx += stride;
        }
    }
}
