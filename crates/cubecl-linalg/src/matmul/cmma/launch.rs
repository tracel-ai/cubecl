use cubecl_core::{
    client::ComputeClient,
    frontend::{Float, TensorArg, TensorHandleRef},
    ir::{Elem, FloatKind},
    tensor_vectorization_factor, Feature, Runtime,
};
use half::f16;

use crate::{
    matmul::cmma::{base::cmma_launch, config::CmmaConfig},
    tensor::{into_contiguous, matrix_layout, MatrixLayout, TensorHandle},
};

use super::config::{TILE_SIZE_K, TILE_SIZE_M, TILE_SIZE_N};

/// Matrix multiplication using [cooperative matrix-multiply and accumulate operations](cubecl_core::cmma).
pub fn matmul_cmma<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, F>,
    rhs: TensorHandle<R, F>,
    out: TensorHandle<R, F>,
    block_config: CmmaConfig,
) -> TensorHandle<R, F> {
    matmul_cmma_ref::<R, F>(
        client,
        lhs.as_ref(),
        rhs.as_ref(),
        out.as_ref(),
        block_config,
    );
    out
}

#[derive(Debug)]
pub enum UnavailabilityReason {
    HighlyPermutatedInput,
    SharedMemoryLimitBusted,
    InvalidConfig(String),
    CmmaInstructionsUnsupported,
}

/// Checks if the matmul cmma can be used.
pub fn check_cmma_availability<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
) -> Result<(), UnavailabilityReason> {
    if !client.features().enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: TILE_SIZE_M,
        k: TILE_SIZE_K,
        n: TILE_SIZE_N,
    }) {
        return Err(UnavailabilityReason::CmmaInstructionsUnsupported);
    }

    Ok(())
}
/// Matrix multiplication using [cooperative matrix-multiply and accumulate operations](cubecl_core::cmma).
pub fn matmul_cmma_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    block_config: CmmaConfig,
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
        (true, true) => matmul_cmma_ref_no_check::<R, F>(client, lhs, rhs, out, block_config),
        (true, false) => matmul_cmma_ref_no_check::<R, F>(
            client,
            lhs,
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            block_config,
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            rhs,
            out,
            block_config,
        ),
        (false, false) => matmul_cmma_ref_no_check::<R, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            block_config,
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

    let m = lhs.shape[rank - 2];
    let k = lhs.shape[rank - 1];
    let n = rhs.shape[rank - 1];

    let available_vectorizations = cmma_config.available_vectorizations();
    let lhs_vectorization =
        tensor_vectorization_factor(&available_vectorizations, lhs.shape, lhs.strides, rank - 1);
    let rhs_vectorization =
        tensor_vectorization_factor(&available_vectorizations, rhs.shape, rhs.strides, rank - 1);
    let out_vectorization =
        tensor_vectorization_factor(&available_vectorizations, out.shape, out.strides, rank - 1);

    unsafe {
        cmma_launch::launch_unchecked::<F, f16, R>(
            client,
            cmma_config.cube_count::<R>(out.shape),
            cmma_config.cube_dim(),
            TensorArg::from_raw_parts(lhs.handle, lhs.strides, lhs.shape, lhs_vectorization),
            TensorArg::from_raw_parts(rhs.handle, rhs.strides, rhs.shape, rhs_vectorization),
            TensorArg::from_raw_parts(out.handle, out.strides, out.shape, out_vectorization),
            cmma_config.comptime_info(m, k, n),
        );
    }
}
