use cubecl_core::{
    client::ComputeClient,
    prelude::{Float, TensorHandleRef},
    Runtime,
};

use crate::tensor::TensorHandle;

use super::kernels::{
    cmma_old::{self, CmmaConfig, PredefinedCmmaConfig},
    matmul,
    tiling2d::{self, Tiling2dConfig},
};

#[derive(Debug)]
pub enum Strategy {
    Accelerated,
    PlaneMma,
    CmmaOld(CmmaConfig),
    Tiling2D(Tiling2dConfig),
}

impl Default for Strategy {
    fn default() -> Self {
        // Still the fastest implementation.
        Strategy::CmmaOld(PredefinedCmmaConfig::M128K16.into())
    }
}

pub fn launch<R: Runtime, EG: Float>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
) {
    match strategy {
        Strategy::Accelerated => matmul::launch(client, lhs, rhs, out, false),
        Strategy::PlaneMma => matmul::launch(client, lhs, rhs, out, true),
        Strategy::CmmaOld(config) => cmma_old::launch(client, lhs, rhs, out, config.clone()),
        Strategy::Tiling2D(config) => tiling2d::launch(client, lhs, rhs, out, config.clone()),
    };
}

pub fn launch_ref<R: Runtime, EG: Float>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<R>,
    rhs: TensorHandleRef<R>,
    out: TensorHandleRef<R>,
) {
    match strategy {
        Strategy::Accelerated => matmul::launch_ref::<R, EG>(client, lhs, rhs, out, false),
        Strategy::PlaneMma => matmul::launch_ref::<R, EG>(client, lhs, rhs, out, true),
        Strategy::CmmaOld(config) => {
            cmma_old::launch_ref::<R, EG>(client, lhs, rhs, out, config.clone())
        }
        Strategy::Tiling2D(config) => {
            tiling2d::launch_ref::<R, EG>(client, lhs, rhs, out, config.clone())
        }
    };
}
