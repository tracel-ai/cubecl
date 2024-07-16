use std::cmp::max;

use cubecl_core::{
    frontend::{Float, TensorArg, F16},
    Compiler, Runtime,
};

use crate::{
    matmul::cmma::{
        base::{cmma_kernel, USE_CMMA},
        config::{cmma_cube_count, cmma_cube_dim, CmmaConfig},
    },
    tensor::{MatrixLayout, Tensor},
};

// Only those values supported at the moment
const BLOCK_SIZE_M: usize = 64;
const BLOCK_SIZE_K: usize = 32;
const BLOCK_SIZE_N: usize = 64;

/// Matrix multiplication using tiling 2d algorithm
pub fn matmul_cmma<R: Runtime, F: Float>(
    lhs: Tensor<R, F>,
    rhs: Tensor<R, F>,
    out: Tensor<R, F>,
    device: &R::Device,
) -> Tensor<R, F> {
    assert!(
        BLOCK_SIZE_K * max(BLOCK_SIZE_M, BLOCK_SIZE_N)
            <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );
    assert!(
        BLOCK_SIZE_M * BLOCK_SIZE_N <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );

    let rank = lhs.rank();
    let m = lhs.shape[rank - 2];
    let k = lhs.shape[rank - 1];
    let n = rhs.shape[rank - 1];

    let client = R::client(device);

    let check_layout = |tensor: Tensor<R, F>| match tensor.matrix_layout() {
        MatrixLayout::Contiguous => (tensor, false),
        MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (tensor, transposed),
        MatrixLayout::HighlyPermuted => {
            panic!("Can't run on highly permuted tensor")
        }
    };
    let (lhs, lhs_transposed) = check_layout(lhs);
    let (rhs, rhs_transposed) = check_layout(rhs);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let lhs_vectorization = match lhs_transposed {
        true => panic!(),
        false => vectorization(k),
    };
    let rhs_vectorization = match rhs_transposed {
        true => 1,
        false => vectorization(n),
    };
    let out_vectorization = vectorization(n);

    let cube_count = cmma_cube_count::<R>(&out.shape, 64, 64);
    let cube_dim = cmma_cube_dim();
    let cube_config = CmmaConfig::new(m, k, n, lhs_transposed, rhs_transposed, USE_CMMA);

    assert!(lhs_vectorization == 4 && rhs_vectorization == 4 && out_vectorization == 4);

    if USE_CMMA {
        cmma_kernel::launch::<F, F16, R>(
            client,
            cube_count,
            cube_dim,
            TensorArg::vectorized(lhs_vectorization, &lhs.handle, &lhs.strides, &lhs.shape),
            TensorArg::vectorized(rhs_vectorization, &rhs.handle, &rhs.strides, &rhs.shape),
            TensorArg::vectorized(out_vectorization, &out.handle, &out.strides, &out.shape),
            cube_config,
        );
    } else {
        cmma_kernel::launch::<F, F, R>(
            client,
            cube_count,
            cube_dim,
            TensorArg::vectorized(lhs_vectorization, &lhs.handle, &lhs.strides, &lhs.shape),
            TensorArg::vectorized(rhs_vectorization, &rhs.handle, &rhs.strides, &rhs.shape),
            TensorArg::vectorized(out_vectorization, &out.handle, &out.strides, &out.shape),
            cube_config,
        );
    }

    out
}
