mod base;
pub mod cmma_matmul;
mod config;
pub(crate) mod launch;
pub mod matmul_batch;
pub mod matmul_global;
pub mod matmul_stage;
pub mod matmul_tile;
pub mod matrix;
pub mod problem;
pub mod stage_dim;

#[cfg(feature = "export_tests")]
pub mod tests;

/// Launch a matrix multiplication kernel.
// pub fn launch_ref<R: Runtime, F: Float>(
//     client: &ComputeClient<R::Server, R::Channel>,
//     lhs: TensorHandleRef<'_, R>,
//     rhs: TensorHandleRef<'_, R>,
//     out: TensorHandleRef<'_, R>,
// ) {
//     let cmma_config: CmmaOldConfig = PredefinedCmmaConfig::M128K16.into();
//     if cmma_available::<R>(client, &cmma_config).is_ok() {
//         cmma_launch_ref::<R, F>(client, lhs, rhs, out, cmma_config);
//     } else {
//         tiling2d::launch_ref::<R, F>(client, lhs, rhs, out, Default::default());
//     }
// }
pub use base::*;
