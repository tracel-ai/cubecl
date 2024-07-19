use std::cmp::max;

use cubecl_core::{
    client::ComputeClient,
    frontend::{Float, TensorArg, TensorHandleRef, F16},
    ir::{Elem, FloatKind},
    Compiler, Feature, Runtime,
};

use crate::{
    matmul::cmma::{
        base::cmma_kernel,
        config::{cmma_cube_count, cmma_cube_dim, CmmaConfig, CmmaLaunchConfig},
    },
    tensor::{matrix_layout, MatrixLayout, TensorHandle},
};

/// Matrix multiplication using [cooperative matrix-multiply and accumulate operations](cubecl_core::cmma).
pub fn matmul_cmma<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, F>,
    rhs: TensorHandle<R, F>,
    out: TensorHandle<R, F>,
) -> TensorHandle<R, F> {
    matmul_cmma_ref::<R, F>(client, lhs.as_ref(), rhs.as_ref(), out.as_ref());
    out
}

#[derive(Debug)]
pub enum UnavailabilityReason {
    TransposedInput, // TODO: Support that case.
    NotMultipleOf4,  // TODO: Support that case.
    HiglyPermutatedInput,
    ShapeMemoryLimitBusted,
    InvalidConfig(String),
    CmmaInstructionsUnsupported,
}

/// Checks if the matmul cmma can be used.
pub fn check_cmma_availability<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    config: Option<&CmmaLaunchConfig>,
) -> Result<(), UnavailabilityReason> {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_layout(tensor.strides) {
        MatrixLayout::Contiguous => Ok(()),
        MatrixLayout::MildlyPermuted {
            transposed: _,
            batch_swap: _,
        } => Err(UnavailabilityReason::TransposedInput),
        MatrixLayout::HighlyPermuted => Err(UnavailabilityReason::HiglyPermutatedInput),
    };

    if !client.features().enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: 16,
        k: 16,
        n: 16,
    }) {
        return Err(UnavailabilityReason::CmmaInstructionsUnsupported);
    }

    check_layout(lhs)?;
    check_layout(rhs)?;

    let rank = lhs.shape.len();
    let m = lhs.shape[rank - 2];
    let k = lhs.shape[rank - 1];
    let n = rhs.shape[rank - 1];

    if !(m % 4 == 0 && k % 4 == 0 && n % 4 == 0) {
        return Err(UnavailabilityReason::NotMultipleOf4);
    }

    if let Some(config) = config {
        let (b_m, b_k, b_n) = (
            config.block_size_m,
            config.block_size_k,
            config.block_size_n,
        );

        if b_k * max(b_m, b_n) > <R::Compiler as Compiler>::max_shared_memory_size() {
            return Err(UnavailabilityReason::ShapeMemoryLimitBusted);
        }

        if b_m * b_n > <R::Compiler as Compiler>::max_shared_memory_size() {
            return Err(UnavailabilityReason::ShapeMemoryLimitBusted);
        }

        if b_k != 2 * config.tile_size {
            return Err(UnavailabilityReason::InvalidConfig(
                "Variable tile number per coop_units not supported".to_string(),
            ));
        }
    }

    Ok(())
}
/// Matrix multiplication using [cooperative matrix-multiply and accumulate operations](cubecl_core::cmma).
pub fn matmul_cmma_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
) {
    let rank = lhs.strides.len();

    let m = lhs.shape[rank - 2];
    let k = lhs.shape[rank - 1];
    let n = rhs.shape[rank - 1];

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

    let cube_count = cmma_cube_count::<R>(out.shape, 64, 64);
    let cube_dim = cmma_cube_dim();
    let launch_config = CmmaLaunchConfig::default();

    cmma_kernel::launch::<F, F16, R>(
        client,
        cube_count,
        cube_dim,
        TensorArg::vectorized(lhs_vectorization, lhs.handle, lhs.strides, lhs.shape),
        TensorArg::vectorized(rhs_vectorization, rhs.handle, rhs.strides, rhs.shape),
        TensorArg::vectorized(out_vectorization, out.handle, out.strides, out.shape),
        CmmaConfig::new(m, k, n, launch_config),
    );
}
