mod base;
pub mod cmma_matmul;
mod config;
mod launch;
pub mod matmul_batch;
pub mod matmul_global;
pub mod matmul_stage;
pub mod matmul_tile;
pub mod matrix;
pub mod problem;
pub mod stage_dim;

#[cfg(feature = "export_tests")]
pub mod tests;

use cubecl_core::prelude::*;

// Launch a matrix multiplication kernel.
pub fn launch_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
) {
    matmul_cmma_ref::<R, F>(client, lhs, rhs, out);
}
pub use base::*;
use launch::matmul_cmma_ref;
