use cubecl_core::{client::ComputeClient, prelude::TensorHandleRef, Runtime};
use cubecl_std::MaybeQuantized;

use crate::tensor::TensorHandle;

use super::{
    components::{
        global::single_stage::{
            WindowDuplicatedLoading, WindowElectedLoading, WindowElectedOnlyLoading,
            WindowSplitPlaneLoading, WindowSplitUnitLoading,
        },
        tile::accelerated::Accelerated,
    },
    kernels::{
        matmul::{
            self, double_buffering::DoubleBufferingAlgorithm, simple::SimpleAlgorithm,
            simple_barrier::SimpleBarrierAlgorithm, simple_pipelined::SimplePipelinedAlgorithm,
            specialized::SpecializedAlgorithm,
        },
        naive,
        tiling2d::{self, Tiling2dConfig},
        MatmulLaunchError,
    },
};

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    Simple,
    SimpleBarrier(SimpleBarrierLoadingStrategy),
    SimplePipelined,
    DoubleBuffering,
    Specialized,
    #[cfg(any(test, feature = "export_tests"))]
    // Very slow, only use for testing.
    PlaneMma,
    Naive,
    Tiling2D(Tiling2dConfig),
    #[default]
    Auto,
}

#[derive(Debug, Clone, Default)]
pub enum SimpleBarrierLoadingStrategy {
    #[default]
    Duplicated,
    Elected,
    ElectedOnly,
    SplitUnit,
    SplitPlane,
}

pub fn launch<R: Runtime, EG: MaybeQuantized>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG::Numeric>,
    rhs: TensorHandle<R, EG::Numeric>,
    out: TensorHandle<R, EG::Numeric>,
) -> Result<(), MatmulLaunchError> {
    launch_ref::<R, EG>(
        strategy,
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
    )
}

pub fn launch_ref<R: Runtime, EG: MaybeQuantized>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<R>,
    rhs: &TensorHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulLaunchError> {
    match strategy {
        Strategy::Simple => {
            matmul::launch_ref::<R, EG, SimpleAlgorithm<Accelerated>>(client, lhs, rhs, out)
        }
        Strategy::SimpleBarrier(loading_strategy) => match loading_strategy {
            SimpleBarrierLoadingStrategy::Duplicated => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, WindowDuplicatedLoading>,
            >(client, lhs, rhs, out),
            SimpleBarrierLoadingStrategy::Elected => {
                matmul::launch_ref::<R, EG, SimpleBarrierAlgorithm<Accelerated, WindowElectedLoading>>(
                    client, lhs, rhs, out,
                )
            }
            SimpleBarrierLoadingStrategy::ElectedOnly => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, WindowElectedOnlyLoading>,
            >(client, lhs, rhs, out),
            SimpleBarrierLoadingStrategy::SplitUnit => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, WindowSplitUnitLoading>,
            >(client, lhs, rhs, out),
            SimpleBarrierLoadingStrategy::SplitPlane => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, WindowSplitPlaneLoading>,
            >(client, lhs, rhs, out),
        },
        Strategy::SimplePipelined => {
            matmul::launch_ref::<R, EG, SimplePipelinedAlgorithm<Accelerated>>(
                client, lhs, rhs, out,
            )
        }
        Strategy::DoubleBuffering => {
            matmul::launch_ref::<R, EG, DoubleBufferingAlgorithm<Accelerated>>(
                client, lhs, rhs, out,
            )
        }
        Strategy::Specialized => {
            matmul::launch_ref::<R, EG, SpecializedAlgorithm<Accelerated>>(client, lhs, rhs, out)
        }
        #[cfg(any(test, feature = "export_tests"))]
        Strategy::PlaneMma => {
            matmul::launch_ref::<R, EG, SimpleAlgorithm<super::components::tile::plane::PlaneMma>>(
                client, lhs, rhs, out,
            )
        }
        Strategy::Tiling2D(config) => {
            tiling2d::launch_ref::<R, EG::Numeric>(client, lhs, rhs, out, config.clone());
            Ok(())
        }
        Strategy::Naive => {
            naive::launch_ref::<R, EG::Numeric>(client, lhs, rhs, out)?;
            Ok(())
        }
        Strategy::Auto => {
            if let Err(err) =
                matmul::launch_ref::<R, EG, SimpleAlgorithm<Accelerated>>(client, lhs, rhs, out)
            {
                match err {
                    super::kernels::MatmulLaunchError::Unavailable(_) => {
                        tiling2d::launch_ref::<R, EG::Numeric>(
                            client,
                            lhs,
                            rhs,
                            out,
                            Tiling2dConfig::default(),
                        )
                    }
                    _ => panic!("{err:?}"),
                }
            }

            Ok(())
        }
    }
}
