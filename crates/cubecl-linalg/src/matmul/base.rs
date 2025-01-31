use cubecl_core::{
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
    Runtime,
};

use crate::tensor::TensorHandle;

use super::{
    components::tile::accelerated::Accelerated,
    kernels::{
        matmul::{self, PipelinedSelector, SpecializedSelector, StandardSelector},
        simple,
        tiling2d::{self, Tiling2dConfig},
        MatmulLaunchError,
    },
};

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    Standard,
    Pipelined,
    Specialized,
    #[cfg(any(test, feature = "export_tests"))]
    // Very slow, only use for testing.
    PlaneMma,
    Simple,
    Tiling2D(Tiling2dConfig),
    #[default]
    Auto,
}

pub fn launch<R: Runtime, In: Numeric, Out: Numeric>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, In>,
    rhs: TensorHandle<R, In>,
    out: TensorHandle<R, Out>,
) -> Result<(), MatmulLaunchError> {
    launch_ref::<R, In, Out>(
        strategy,
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
    )
}

pub fn launch_ref<R: Runtime, In: Numeric, Out: Numeric>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<R>,
    rhs: &TensorHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulLaunchError> {
    match strategy {
        Strategy::Standard => {
            matmul::launch_ref::<R, In, Out, StandardSelector<Accelerated>>(client, lhs, rhs, out)
        }
        Strategy::Pipelined => {
            matmul::launch_ref::<R, In, Out, PipelinedSelector<Accelerated>>(client, lhs, rhs, out)
        }
        Strategy::Specialized => {
            matmul::launch_ref::<R, In, Out, SpecializedSelector<Accelerated>>(
                client, lhs, rhs, out,
            )
        }
        #[cfg(any(test, feature = "export_tests"))]
        Strategy::PlaneMma => matmul::launch_ref::<
            R,
            In,
            Out,
            StandardSelector<super::components::tile::plane::PlaneMma>,
        >(client, lhs, rhs, out),
        Strategy::Tiling2D(config) => {
            tiling2d::launch_ref::<R, In>(client, lhs, rhs, out, config.clone());
            Ok(())
        }
        Strategy::Simple => {
            simple::launch_ref::<R, In>(client, lhs, rhs, out)?;
            Ok(())
        }
        Strategy::Auto => {
            if let Err(err) = matmul::launch_ref::<R, In, Out, StandardSelector<Accelerated>>(
                client, lhs, rhs, out,
            ) {
                match err {
                    super::kernels::MatmulLaunchError::Unavailable(_) => {
                        tiling2d::launch_ref::<R, In>(
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
