use cubecl_core::{client::ComputeClient, prelude::Float, Runtime};

use crate::tensor::TensorHandle;

use super::kernels::{
    cmma_matmul,
    cmma_old::{self, CmmaConfig},
    tiling2d::{self, Tiling2dConfig},
};

pub enum Strategy {
    Accelerated,
    PlaneMma,
    CmmaOld(CmmaConfig),
    Tiling2D(Tiling2dConfig),
}

/// TODO should be numeric
pub fn launch<R: Runtime, EG: Float>(
    strategy: Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
) {
    match strategy {
        Strategy::Accelerated => cmma_matmul::launch(client, lhs, rhs, out, false),
        Strategy::PlaneMma => cmma_matmul::launch(client, lhs, rhs, out, true),
        Strategy::CmmaOld(config) => cmma_old::launch(client, lhs, rhs, out, config),
        Strategy::Tiling2D(config) => tiling2d::launch(client, lhs, rhs, out, config),
    };
}
