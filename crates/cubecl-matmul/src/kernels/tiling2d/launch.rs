use std::cmp::max;

use cubecl_core::prelude::*;

use crate::kernels::tiling2d::{
    base::tiling2d_cube_kernel,
    config::{CubeTiling2dConfig, tiling2d_cube_count, tiling2d_cube_dim},
};
use cubecl_std::tensor::{MatrixBatchLayout, TensorHandle, into_contiguous, matrix_batch_layout};

use super::config::Tiling2dConfig;

/// Matrix multiplication using tiling 2d algorithm.
pub fn matmul_tiling_2d<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, F>,
    rhs: TensorHandle<R, F>,
    out: TensorHandle<R, F>,
    config: Tiling2dConfig,
) -> TensorHandle<R, F> {
    matmul_tiling_2d_ref::<R, F>(client, &lhs.as_ref(), &rhs.as_ref(), &out.as_ref(), config);

    out
}

/// Matrix multiplication using tiling 2d algorithm.
pub fn matmul_tiling_2d_ref<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    config: Tiling2dConfig,
) {
    assert!(
        EG::size().unwrap() * config.block_size_k * max(config.block_size_m, config.block_size_n)
            <= client.properties().hardware.max_shared_memory_size,
        "Shared memory limit will be busted. "
    );
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_batch_layout(tensor.strides) {
        MatrixBatchLayout::Contiguous => true,
        MatrixBatchLayout::MildlyPermuted {
            transposed: _,
            batch_swap: _,
        } => true,
        MatrixBatchLayout::HighlyPermuted => false,
    };
    let lhs_correct_layout = check_layout(lhs);
    let rhs_correct_layout = check_layout(rhs);

    match (lhs_correct_layout, rhs_correct_layout) {
        (true, true) => matmul_tiling_2d_ref_no_check::<R, EG>(client, lhs, rhs, out, config),
        (true, false) => matmul_tiling_2d_ref_no_check::<R, EG>(
            client,
            lhs,
            &into_contiguous::<R, EG>(client, rhs).as_ref(),
            out,
            config,
        ),
        (false, true) => matmul_tiling_2d_ref_no_check::<R, EG>(
            client,
            &into_contiguous::<R, EG>(client, lhs).as_ref(),
            rhs,
            out,
            config,
        ),
        (false, false) => matmul_tiling_2d_ref_no_check::<R, EG>(
            client,
            &into_contiguous::<R, EG>(client, lhs).as_ref(),
            &into_contiguous::<R, EG>(client, rhs).as_ref(),
            out,
            config,
        ),
    }
}

/// Matrix multiplication using tiling 2d algorithm.
fn matmul_tiling_2d_ref_no_check<R: Runtime, N: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    config: Tiling2dConfig,
) {
    let rank = lhs.strides.len();

    let m = lhs.shape[rank - 2];
    let k = lhs.shape[rank - 1];
    let n = rhs.shape[rank - 1];

    let check_layout = |strides: &[usize]| match matrix_batch_layout(strides) {
        MatrixBatchLayout::Contiguous => false,
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => transposed,
        MatrixBatchLayout::HighlyPermuted => {
            panic!("Can't run on highly permuted tensor")
        }
    };
    let lhs_transposed = check_layout(lhs.strides);
    let rhs_transposed = check_layout(rhs.strides);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let lhs_vectorization = match lhs_transposed {
        true => vectorization(m),
        false => 1,
    };
    let rhs_vectorization = match rhs_transposed {
        true => 1,
        false => vectorization(n),
    };
    let out_vectorization = vectorization(n);

    let cube_count = tiling2d_cube_count(out.shape, &config);
    let cube_dim = tiling2d_cube_dim(&config);
    let cube_config = CubeTiling2dConfig::new(&config, m, k, n, lhs_transposed, rhs_transposed);

    unsafe {
        tiling2d_cube_kernel::launch_unchecked::<N, R>(
            client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts::<N>(lhs.handle, lhs.strides, lhs.shape, lhs_vectorization),
            TensorArg::from_raw_parts::<N>(rhs.handle, rhs.strides, rhs.shape, rhs_vectorization),
            TensorArg::from_raw_parts::<N>(out.handle, out.strides, out.shape, out_vectorization),
            cube_config,
        );
    }
}
