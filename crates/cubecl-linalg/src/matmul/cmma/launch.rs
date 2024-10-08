use cubecl_core::{
    client::ComputeClient,
    frontend::{Float, TensorArg, TensorHandleRef},
    tensor_line_size, Runtime,
};
use half::f16;

use crate::{
    matmul::cmma::{base::cmma_launch, config::CmmaConfig},
    tensor::{into_contiguous, matrix_layout, MatrixLayout, TensorHandle},
};

/// Matrix multiplication using [cooperative matrix-multiply and accumulate operations](cubecl_core::cmma).
pub fn matmul_cmma<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, F>,
    rhs: TensorHandle<R, F>,
    out: TensorHandle<R, F>,
    cmma_config: CmmaConfig,
) -> TensorHandle<R, F> {
    matmul_cmma_ref::<R, F>(
        client,
        lhs.as_ref(),
        rhs.as_ref(),
        out.as_ref(),
        cmma_config,
    );
    out
}

/// Matrix multiplication using [cooperative matrix-multiply and accumulate operations](cubecl_core::cmma).
pub fn matmul_cmma_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    cmma_config: CmmaConfig,
) {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_layout(tensor.strides) {
        MatrixLayout::Contiguous => true,
        MatrixLayout::MildlyPermuted {
            transposed: _,
            batch_swap: _,
        } => false,
        MatrixLayout::HighlyPermuted => false,
    };

    let lhs_correct_layout = check_layout(&lhs);
    let rhs_correct_layout = check_layout(&rhs);

    match (lhs_correct_layout, rhs_correct_layout) {
        (true, true) => matmul_cmma_ref_no_check::<R, F>(client, lhs, rhs, out, cmma_config),
        (true, false) => matmul_cmma_ref_no_check::<R, F>(
            client,
            lhs,
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            cmma_config,
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            rhs,
            out,
            cmma_config,
        ),
        (false, false) => matmul_cmma_ref_no_check::<R, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            cmma_config,
        ),
    }
}

fn matmul_cmma_ref_no_check<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    cmma_config: CmmaConfig,
) {
    let rank = lhs.strides.len();

    let m = lhs.shape[rank - 2] as u32;
    let k = lhs.shape[rank - 1] as u32;
    let n = rhs.shape[rank - 1] as u32;

    let available_vectorizations = R::supported_line_sizes();
    let lhs_vectorization =
        tensor_line_size(available_vectorizations, lhs.shape, lhs.strides, rank - 1);
    let rhs_vectorization =
        tensor_line_size(available_vectorizations, rhs.shape, rhs.strides, rank - 1);
    let out_vectorization =
        tensor_line_size(available_vectorizations, out.shape, out.strides, rank - 1);

    unsafe {
        cmma_launch::launch_unchecked::<F, f16, R>(
            client,
            cmma_config.cube_count(out.shape),
            cmma_config.cube_dim(),
            TensorArg::from_raw_parts(lhs.handle, lhs.strides, lhs.shape, lhs_vectorization),
            TensorArg::from_raw_parts(rhs.handle, rhs.strides, rhs.shape, rhs_vectorization),
            TensorArg::from_raw_parts(out.handle, out.strides, out.shape, out_vectorization),
            cmma_config.comptime_info(m, k, n),
        );
    }
}
