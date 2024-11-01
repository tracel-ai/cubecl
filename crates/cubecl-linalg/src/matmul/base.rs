use cubecl_core::{
    client::ComputeClient,
    prelude::{Float, TensorHandleRef},
    Runtime,
};

use crate::tensor::TensorHandle;

use super::kernels::{
    cmma_matmul,
    cmma_old::{self, config::PredefinedCmmaConfig, is_available, CmmaConfig},
    tiling2d::{self, Tiling2dConfig},
};

#[derive(Debug)]
pub enum Strategy {
    Accelerated,
    PlaneMma,
    CmmaOld(CmmaConfig),
    Tiling2D(Tiling2dConfig),
}

pub fn launch<R: Runtime, EG: Float>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
) {
    match strategy {
        Strategy::Accelerated => cmma_matmul::launch(client, lhs, rhs, out, false),
        Strategy::PlaneMma => cmma_matmul::launch(client, lhs, rhs, out, true),
        Strategy::CmmaOld(config) => cmma_old::launch(client, lhs, rhs, out, config.clone()),
        Strategy::Tiling2D(config) => tiling2d::launch(client, lhs, rhs, out, config.clone()),
    };
}

pub fn launch_ref<R: Runtime, EG: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<R>,
    rhs: TensorHandleRef<R>,
    out: TensorHandleRef<R>,
) {
    let cmma_config = PredefinedCmmaConfig::M128K16.into();

    match is_available::<R, EG>(client, &cmma_config) {
        Ok(_) => cmma_old::launch_ref::<R, EG>(client, lhs, rhs, out, cmma_config),
        Err(_) => tiling2d::launch_ref::<R, EG>(client, lhs, rhs, out, Default::default()),
    }
}
