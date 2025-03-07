use cubecl_core::{client::ComputeClient, prelude::TensorHandleRef, Runtime};
use cubecl_std::MaybeQuantized;

use crate::tensor::TensorHandle;

use super::{
    components::{
        global::loader::{
            r#async::{
                CyclicWindowLoading, MaximizeSliceLengthLoading, MaximizeUnitCountLoading,
                WindowCooperativeLoading,
            },
            sync::StridedCoalescedLoading,
        },
        stage::ColMajorTilingOrder,
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
    Simple(SyncLoadingStrategy),
    SimpleBarrier(AsyncLoadingStrategy),
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

#[derive(Debug, Clone)]
pub enum SyncLoadingStrategy {
    Cyclic,
    Strided,
}

#[derive(Debug, Clone)]
pub enum AsyncLoadingStrategy {
    Cooperative,
    Cyclic,
    MaximizeSliceLength,
    MaximizeUnitCount,
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
        Strategy::Simple(loading_strategy) => match loading_strategy {
            SyncLoadingStrategy::Cyclic => {
                matmul::launch_ref::<R, EG, SimpleAlgorithm<Accelerated>>(client, lhs, rhs, out)
            }
            SyncLoadingStrategy::Strided => matmul::launch_ref::<
                R,
                EG,
                SimpleAlgorithm<Accelerated, StridedCoalescedLoading, StridedCoalescedLoading>,
            >(client, lhs, rhs, out),
        },
        Strategy::SimpleBarrier(loading_strategy) => match loading_strategy {
            AsyncLoadingStrategy::Cooperative => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, WindowCooperativeLoading>,
            >(client, lhs, rhs, out),
            AsyncLoadingStrategy::Cyclic => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, CyclicWindowLoading<ColMajorTilingOrder>>,
            >(client, lhs, rhs, out),
            AsyncLoadingStrategy::MaximizeSliceLength => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, MaximizeSliceLengthLoading>,
            >(client, lhs, rhs, out),
            AsyncLoadingStrategy::MaximizeUnitCount => matmul::launch_ref::<
                R,
                EG,
                SimpleBarrierAlgorithm<Accelerated, MaximizeUnitCountLoading>,
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
