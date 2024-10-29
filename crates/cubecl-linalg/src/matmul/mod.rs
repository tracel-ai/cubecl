mod base;
pub mod cmma_matmul;
mod config;
pub mod matmul_batch;
pub mod matmul_global;
pub mod matmul_stage;
pub mod matmul_tile;
pub mod matrix;
pub mod problem;
pub mod stage_dim;

#[cfg(feature = "export_tests")]
pub mod tests;

use cmma_matmul::{
    launch::{matmul_cmma_ref, CmmaLaunchDispatch, PlaneMmaLaunchDispatch},
    tile::check_cmma_availability,
};
use cubecl_core::prelude::*;

// Launch a matrix multiplication kernel.
pub fn launch_ref<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    disable_cmma: bool,
) {
    if !disable_cmma && check_cmma_availability::<R>(client).is_ok() {
        matmul_cmma_ref::<R, EG, CmmaLaunchDispatch>(client, lhs, rhs, out);
    } else {
        matmul_cmma_ref::<R, EG, PlaneMmaLaunchDispatch>(client, lhs, rhs, out);
    }
}

pub fn launch<R: Runtime, E: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, E>,
    rhs: TensorHandle<R, E>,
    out: TensorHandle<R, E>,
    disable_cmma: bool,
) -> TensorHandle<R, E> {
    launch_ref::<R, E>(
        client,
        lhs.as_ref(),
        rhs.as_ref(),
        out.as_ref(),
        disable_cmma,
    );
    out
}

pub use base::*;

use crate::tensor::TensorHandle;
