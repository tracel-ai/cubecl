use cmma_old::config::{CmmaOldConfig, PredefinedCmmaConfig};
use cmma_old::{is_available as cmma_available, launch_ref as cmma_launch_ref};
use cubecl_core::prelude::*;

mod base;
pub mod cmma_matmul;
pub mod cmma_old;
pub(crate) mod launch;
pub mod matmul_batch;
pub mod matmul_global;
pub mod matmul_stage;
pub mod matmul_tile;
pub mod matrix_layout;
pub mod problem;
pub mod stage_info;
mod config;

/// Contains algorithms for tiling 2d matrix multiplication when cooperative matrix are not
/// available.
pub mod tiling2d;

#[cfg(feature = "export_tests")]
pub mod tests;

/// Launch a matrix multiplication kernel.
pub fn launch_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
) {
    let cmma_config: CmmaOldConfig = PredefinedCmmaConfig::M128K16.into();
    if cmma_available::<R>(client, &cmma_config).is_ok() {
        cmma_launch_ref::<R, F>(client, lhs, rhs, out, cmma_config);
    } else {
        tiling2d::launch_ref::<R, F>(client, lhs, rhs, out, Default::default());
    }
}

pub use base::*;
