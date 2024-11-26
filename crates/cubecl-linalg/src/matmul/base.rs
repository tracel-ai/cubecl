use cubecl_core::{
    client::ComputeClient,
    prelude::{Float, TensorHandleRef},
    Runtime,
};

use crate::tensor::TensorHandle;

use super::kernels::{
    cmma_old::{self, CmmaConfig},
    matmul, simple,
    tiling2d::{self, Tiling2dConfig},
};

#[derive(Debug, Clone)]
pub enum Strategy {
    Accelerated,
    PlaneMma,
    Simple,
    CmmaOld(CmmaConfig),
    Tiling2D(Tiling2dConfig),
}

impl Default for Strategy {
    fn default() -> Self {
        Strategy::Accelerated
    }
}

pub fn launch<R: Runtime, EG: Float>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
) {
    launch_ref::<R, EG>(strategy, client, lhs.as_ref(), rhs.as_ref(), out.as_ref());
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
        Strategy::Simple => simple::launch_ref::<R, EG>(client, lhs, rhs, out),
    };
}
