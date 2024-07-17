use std::cmp::max;

use cubecl_core::{
    frontend::{Float, TensorArg, F16},
    Compiler, Runtime,
};

use crate::{
    matmul::cmma::{
        base::cmma_kernel,
        config::{cmma_cube_count, cmma_cube_dim, CmmaConfig},
    },
    tensor::{MatrixLayout, Tensor},
};

/// Matrix multiplication using tiling 2d algorithm
pub fn matmul_cmma<R: Runtime, F: Float>(
    lhs: Tensor<R, F>,
    rhs: Tensor<R, F>,
    out: Tensor<R, F>,
    device: &R::Device,
) -> Tensor<R, F> {
    let rank = lhs.rank();
    let m = lhs.shape[rank - 2];
    let k = lhs.shape[rank - 1];
    let n = rhs.shape[rank - 1];

    let client = R::client(device);

    let check_layout = |tensor: &Tensor<R, F>| match tensor.matrix_layout() {
        MatrixLayout::Contiguous => {}
        MatrixLayout::MildlyPermuted {
            transposed: _,
            batch_swap: _,
        } => panic!("Transposed input not supported yet."),
        MatrixLayout::HighlyPermuted => {
            panic!("Can't run on highly permuted tensor.")
        }
    };
    check_layout(&lhs);
    check_layout(&rhs);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let lhs_vectorization = vectorization(k);
    let rhs_vectorization = vectorization(n);
    let out_vectorization = vectorization(n);

    let cube_count = cmma_cube_count::<R>(&out.shape, 64, 64);
    let cube_dim = cmma_cube_dim();
    let config = CmmaConfig::default();
    let (b_m, b_k, b_n) = (
        config.block_size_m.val as usize,
        config.block_size_k.val as usize,
        config.block_size_n.val as usize,
    );

    assert!(
        lhs_vectorization == 4 && rhs_vectorization == 4 && out_vectorization == 4,
        "Only vec4 is supported"
    );
    assert!(
        m % b_m == 0 && k % b_k == 0 && n % b_n == 0,
        "Check bounds not supported yet. "
    );
    assert!(
        b_k * max(b_m, b_n) <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );
    assert!(
        b_m * b_n <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );
    assert!(
        b_k == 2 * config.tile_size.val as usize,
        "Variable tile number per coop_units not supported"
    );

    cmma_kernel::launch::<F, F16, R>(
        client,
        cube_count,
        cube_dim,
        TensorArg::vectorized(lhs_vectorization, &lhs.handle, &lhs.strides, &lhs.shape),
        TensorArg::vectorized(rhs_vectorization, &rhs.handle, &rhs.strides, &rhs.shape),
        TensorArg::vectorized(out_vectorization, &out.handle, &out.strides, &out.shape),
        config,
    );

    out
}
