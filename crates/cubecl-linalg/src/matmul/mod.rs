use cmma::config::{CmmaConfig, PredefinedCmmaConfig};
use cubecl_core::prelude::*;

/// Contains algorithms for cooperative matrix multiplication.
pub mod cmma;

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
    let cmma_config: CmmaConfig = PredefinedCmmaConfig::M128K16.into();
    if cmma::is_available::<R>(client, &cmma_config).is_ok() {
        cmma::launch_ref::<R, F>(client, lhs, rhs, out, cmma_config);
    } else {
        tiling2d::launch_ref::<R, F>(client, lhs, rhs, out, Default::default());
    }
}
