mod base;
pub mod batch;
pub mod cmma_matmul;
mod config;
pub mod global;
pub mod matrix;
pub mod problem;
pub mod stage;
pub mod stage_dim;
pub mod tile;

pub use base::*;

use crate::tensor::TensorHandle;

use cmma_matmul::launch::{matmul_cmma_ref, CmmaLaunchDispatch, PlaneMmaLaunchDispatch};
use cubecl_core::prelude::*;

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
pub fn launch_ref<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    disable_cmma: bool,
) {
    if !disable_cmma && tile::accelerated::check_availability::<R>(client).is_ok() {
        matmul_cmma_ref::<R, EG, CmmaLaunchDispatch>(client, lhs, rhs, out);
    } else {
        matmul_cmma_ref::<R, EG, PlaneMmaLaunchDispatch>(client, lhs, rhs, out);
    }
}

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
pub fn launch<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
    disable_cmma: bool,
) -> TensorHandle<R, EG> {
    launch_ref::<R, EG>(
        client,
        lhs.as_ref(),
        rhs.as_ref(),
        out.as_ref(),
        disable_cmma,
    );
    out
}
