use cubecl_core::{
    client::ComputeClient,
    prelude::{Float, TensorHandleRef},
    Runtime,
};

use crate::tensor::TensorHandle;

use super::kernels::{
    matmul, simple,
    tiling2d::{self, Tiling2dConfig},
    MatmulLaunchError,
};

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    Accelerated,
    PlaneMma,
    Simple,
    Tiling2D(Tiling2dConfig),
    #[default]
    Auto,
}

pub fn launch<R: Runtime, EG: Float>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
) -> Result<(), MatmulLaunchError> {
    launch_ref::<R, EG>(
        strategy,
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
    )
}

pub fn launch_ref<R: Runtime, EG: Float>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<R>,
    rhs: &TensorHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulLaunchError> {
    match strategy {
        Strategy::Accelerated => matmul::launch_ref::<R, EG>(client, lhs, rhs, out, false),
        // Strategy::PlaneMma => matmul::launch_ref::<R, EG>(client, lhs, rhs, out, true),
        Strategy::Tiling2D(config) => {
            tiling2d::launch_ref::<R, EG>(client, lhs, rhs, out, config.clone());
            Ok(())
        }
        _ => todo!(),
        // Strategy::Simple => {
        //     simple::launch_ref::<R, EG>(client, lhs, rhs, out);
        //     Ok(())
        // }
        // Strategy::Auto => {
        //     if let Err(err) = matmul::launch_ref::<R, EG>(client, lhs, rhs, out, false) {
        //         match err {
        //             super::kernels::MatmulLaunchError::Unavailable(_) => {
        //                 tiling2d::launch_ref::<R, EG>(
        //                     client,
        //                     lhs,
        //                     rhs,
        //                     out,
        //                     Tiling2dConfig::default(),
        //                 )
        //             }
        //             _ => panic!("{err:?}"),
        //         }
        //     }

        //     Ok(())
        // }
    }
}
