use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use crate::tensor::TensorHandle;

use super::{
    components::{
        MatmulPrecision,
        global::load::{
            async_full_cooperative, async_full_cyclic, async_full_maximize_slice_length,
            async_full_maximize_unit_count, sync_full_strided, sync_full_tilewise,
        },
        stage::{ColMajorTilingOrder, RowMajorTilingOrder},
        tile::accelerated_matmul::AcceleratedMatmul,
    },
    kernels::{
        MatmulLaunchError,
        matmul::{
            self,
            double_buffering::{
                CyclicDoubleBufferingAlgorithm, HybridDoubleBufferingAlgorithm,
                TilewiseDoubleBufferingAlgorithm,
            },
            simple::SimpleAlgorithm,
            simple_barrier::SimpleBarrierAlgorithm,
            simple_tma::SimpleTmaAlgorithm,
        },
        naive,
        tiling2d::{self, Tiling2dConfig},
    },
};

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    Simple(SyncLoadingStrategy),
    SimpleBarrier(AsyncLoadingStrategy),
    DoubleBuffering(SyncBufferLoadingStrategy),
    Naive,
    Tiling2D(Tiling2dConfig),
    #[default]
    Auto,
}

#[derive(Debug, Clone)]
pub enum SyncLoadingStrategy {
    Cyclic,
    Strided,
    Tilewise,
}

#[derive(Debug, Clone)]
pub enum SyncBufferLoadingStrategy {
    Cyclic,
    Tilewise,
    Hybrid,
}

#[derive(Debug, Clone)]
pub enum AsyncLoadingStrategy {
    Cooperative,
    Cyclic,
    MaximizeSliceLength,
    MaximizeUnitCount,
    Tma,
}

#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, MP: MatmulPrecision>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, MP::EI>,
    rhs: TensorHandle<R, MP::EI>,
    out: TensorHandle<R, MP::EO>,
) -> Result<(), MatmulLaunchError> {
    launch_ref::<R, MP>(
        strategy,
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
    )
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, MP: MatmulPrecision>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<R>,
    rhs: &TensorHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulLaunchError> {
    match strategy {
        Strategy::Simple(loading_strategy) => match loading_strategy {
            SyncLoadingStrategy::Cyclic => {
                matmul::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(client, lhs, rhs, out)
            }
            SyncLoadingStrategy::Strided => matmul::launch_ref::<
                R,
                MP,
                SimpleAlgorithm<
                    AcceleratedMatmul,
                    sync_full_strided::LoadingStrategy,
                    sync_full_strided::LoadingStrategy,
                >,
            >(client, lhs, rhs, out),
            SyncLoadingStrategy::Tilewise => matmul::launch_ref::<
                R,
                MP,
                SimpleAlgorithm<
                    AcceleratedMatmul,
                    sync_full_tilewise::LoadingStrategy<ColMajorTilingOrder>,
                    sync_full_tilewise::LoadingStrategy<RowMajorTilingOrder>,
                >,
            >(client, lhs, rhs, out),
        },
        Strategy::SimpleBarrier(loading_strategy) => match loading_strategy {
            AsyncLoadingStrategy::Cooperative => matmul::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<AcceleratedMatmul, async_full_cooperative::LoadingStrategy>,
            >(client, lhs, rhs, out),
            AsyncLoadingStrategy::Cyclic => matmul::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
                >,
            >(client, lhs, rhs, out),
            AsyncLoadingStrategy::MaximizeSliceLength => matmul::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_maximize_slice_length::LoadingStrategy,
                >,
            >(client, lhs, rhs, out),
            AsyncLoadingStrategy::MaximizeUnitCount => matmul::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_maximize_unit_count::LoadingStrategy,
                >,
            >(client, lhs, rhs, out),
            AsyncLoadingStrategy::Tma => matmul::matmul_cmma_tma_ref_no_check::<
                R,
                MP,
                SimpleTmaAlgorithm<AcceleratedMatmul>,
            >(client, lhs, rhs, out, (false, false)),
        },
        Strategy::DoubleBuffering(loading_strategy) => match loading_strategy {
            SyncBufferLoadingStrategy::Cyclic => {
                matmul::launch_ref::<R, MP, CyclicDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, rhs, out,
                )
            }
            SyncBufferLoadingStrategy::Tilewise => {
                matmul::launch_ref::<R, MP, TilewiseDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, rhs, out,
                )
            }
            SyncBufferLoadingStrategy::Hybrid => {
                matmul::launch_ref::<R, MP, HybridDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, rhs, out,
                )
            }
        },
        Strategy::Tiling2D(config) => {
            // TODO Implement tiling2d with EI and EO
            tiling2d::launch_ref::<R, MP::EI>(client, lhs, rhs, out, config.clone());
            Ok(())
        }
        Strategy::Naive => {
            // TODO Implement naive with EI and EO
            naive::launch_ref::<R, MP::EI>(client, lhs, rhs, out)?;
            Ok(())
        }
        Strategy::Auto => {
            if let Err(err) =
                matmul::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(client, lhs, rhs, out)
            {
                match err {
                    super::kernels::MatmulLaunchError::Unavailable(_) => {
                        // TODO Implement naive with EI and EO
                        tiling2d::launch_ref::<R, MP::EI>(
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
