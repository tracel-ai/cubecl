use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::global::load::{sync_buffer_cyclic, sync_buffer_tilewise},
    kernels::matmul::double_unit::DoubleUnitAlgorithm,
};

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
            ordered_double_buffering::OrderedDoubleBufferingAlgorithm,
            simple::SimpleAlgorithm,
            simple_barrier::SimpleBarrierAlgorithm,
            simple_tma::SimpleTmaAlgorithm,
            simple_unit::SimpleUnitAlgorithm,
        },
        naive,
        tiling2d::{self, Tiling2dConfig},
    },
};

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    Simple(SyncLoadingStrategy),
    SimpleBarrier(AsyncLoadingStrategy),
    DoubleBuffering(SyncBufferLoadingStrategy, TileMatmulStrategy),
    SimpleUnit,
    OrderedDoubleBuffering,
    Naive,
    Tiling2D(Tiling2dConfig),
    #[default]
    Auto,
}

#[derive(Debug, Clone, Default)]
pub enum TileMatmulStrategy {
    /// Uses tensor cores instructions.
    Accelerated,
    /// Uses normal instructions.
    Register,
    /// Start with tensor cores and fallback on units if no tensor cores instructions is supported.
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
    lhs_scale: Option<TensorHandle<R, f32>>,
    rhs: TensorHandle<R, MP::EI>,
    rhs_scale: Option<TensorHandle<R, f32>>,
    out: TensorHandle<R, MP::EO>,
) -> Result<(), MatmulLaunchError> {
    launch_ref::<R, MP>(
        strategy,
        client,
        &lhs.as_ref(),
        &lhs_scale.as_ref().map(|it| it.as_ref()),
        &rhs.as_ref(),
        &rhs_scale.as_ref().map(|it| it.as_ref()),
        &out.as_ref(),
    )
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, MP: MatmulPrecision>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<R>,
    lhs_scale: &Option<TensorHandleRef<R>>,
    rhs: &TensorHandleRef<R>,
    rhs_scale: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulLaunchError> {
    match strategy {
        Strategy::Simple(loading_strategy) => match loading_strategy {
            SyncLoadingStrategy::Cyclic => {
                matmul::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(
                    client, lhs, lhs_scale, rhs, rhs_scale, out,
                )
            }
            SyncLoadingStrategy::Strided => {
                matmul::launch_ref::<
                    R,
                    MP,
                    SimpleAlgorithm<
                        AcceleratedMatmul,
                        sync_full_strided::LoadingStrategy,
                        sync_full_strided::LoadingStrategy,
                    >,
                >(client, lhs, lhs_scale, rhs, rhs_scale, out)
            }
            SyncLoadingStrategy::Tilewise => {
                matmul::launch_ref::<
                    R,
                    MP,
                    SimpleAlgorithm<
                        AcceleratedMatmul,
                        sync_full_tilewise::LoadingStrategy<ColMajorTilingOrder>,
                        sync_full_tilewise::LoadingStrategy<RowMajorTilingOrder>,
                    >,
                >(client, lhs, lhs_scale, rhs, rhs_scale, out)
            }
        },
        Strategy::SimpleBarrier(loading_strategy) => match loading_strategy {
            AsyncLoadingStrategy::Cooperative => matmul::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<AcceleratedMatmul, async_full_cooperative::LoadingStrategy>,
            >(
                client, lhs, lhs_scale, rhs, rhs_scale, out
            ),
            AsyncLoadingStrategy::Cyclic => {
                matmul::launch_ref::<
                    R,
                    MP,
                    SimpleBarrierAlgorithm<
                        AcceleratedMatmul,
                        async_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
                    >,
                >(client, lhs, lhs_scale, rhs, rhs_scale, out)
            }
            AsyncLoadingStrategy::MaximizeSliceLength => {
                matmul::launch_ref::<
                    R,
                    MP,
                    SimpleBarrierAlgorithm<
                        AcceleratedMatmul,
                        async_full_maximize_slice_length::LoadingStrategy,
                    >,
                >(client, lhs, lhs_scale, rhs, rhs_scale, out)
            }
            AsyncLoadingStrategy::MaximizeUnitCount => {
                matmul::launch_ref::<
                    R,
                    MP,
                    SimpleBarrierAlgorithm<
                        AcceleratedMatmul,
                        async_full_maximize_unit_count::LoadingStrategy,
                    >,
                >(client, lhs, lhs_scale, rhs, rhs_scale, out)
            }
            AsyncLoadingStrategy::Tma => {
                matmul::matmul_cmma_tma_ref_no_check::<R, MP, SimpleTmaAlgorithm<AcceleratedMatmul>>(
                    client,
                    lhs,
                    lhs_scale,
                    rhs,
                    rhs_scale,
                    out,
                    (false, false),
                )
            }
        },
        Strategy::DoubleBuffering(loading_strategy, tile_strategy) => {
            launch_double_buffering::<R, MP>(
                loading_strategy,
                tile_strategy,
                client,
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                out,
            )
        }
        Strategy::OrderedDoubleBuffering => {
            matmul::launch_ref::<R, MP, OrderedDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                client, lhs, lhs_scale, rhs, rhs_scale, out,
            )
        }
        Strategy::SimpleUnit => matmul::launch_ref::<R, MP, SimpleUnitAlgorithm>(
            client, lhs, lhs_scale, rhs, rhs_scale, out,
        ),
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
            if let Err(err) = matmul::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(
                client, lhs, lhs_scale, rhs, rhs_scale, out,
            ) {
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

fn launch_double_buffering<R: Runtime, MP: MatmulPrecision>(
    strategy: &SyncBufferLoadingStrategy,
    tile_strategy: &TileMatmulStrategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<R>,
    lhs_scale: &Option<TensorHandleRef<R>>,
    rhs: &TensorHandleRef<R>,
    rhs_scale: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulLaunchError> {
    let cyclic_acc = || {
        matmul::launch_ref::<R, MP, CyclicDoubleBufferingAlgorithm<AcceleratedMatmul>>(
            client, lhs, lhs_scale, rhs, rhs_scale, out,
        )
    };
    let cyclic_reg = || {
        matmul::launch_ref::<
            R,
            MP,
            DoubleUnitAlgorithm<
                sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
                sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
            >,
        >(client, lhs, lhs_scale, rhs, rhs_scale, out)
    };
    let tilewise_acc = || {
        matmul::launch_ref::<R, MP, TilewiseDoubleBufferingAlgorithm<AcceleratedMatmul>>(
            client, lhs, lhs_scale, rhs, rhs_scale, out,
        )
    };
    let tilewise_reg = || {
        matmul::launch_ref::<
            R,
            MP,
            DoubleUnitAlgorithm<
                sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
                sync_buffer_tilewise::LoadingStrategy<ColMajorTilingOrder>,
            >,
        >(client, lhs, lhs_scale, rhs, rhs_scale, out)
    };
    let hybrid_acc = || {
        matmul::launch_ref::<
            R,
            MP,
            DoubleUnitAlgorithm<
                sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
                sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
            >,
        >(client, lhs, lhs_scale, rhs, rhs_scale, out)
    };
    let hybrid_reg = || {
        matmul::launch_ref::<R, MP, HybridDoubleBufferingAlgorithm<AcceleratedMatmul>>(
            client, lhs, lhs_scale, rhs, rhs_scale, out,
        )
    };

    match strategy {
        SyncBufferLoadingStrategy::Cyclic => match tile_strategy {
            TileMatmulStrategy::Accelerated => cyclic_acc(),
            TileMatmulStrategy::Register => cyclic_reg(),
            TileMatmulStrategy::Auto => match cyclic_acc() {
                Ok(val) => Ok(val),
                Err(_) => cyclic_reg(),
            },
        },
        SyncBufferLoadingStrategy::Tilewise => match tile_strategy {
            TileMatmulStrategy::Accelerated => tilewise_acc(),
            TileMatmulStrategy::Register => tilewise_reg(),
            TileMatmulStrategy::Auto => match tilewise_acc() {
                Ok(val) => Ok(val),
                Err(_) => tilewise_reg(),
            },
        },
        SyncBufferLoadingStrategy::Hybrid => match tile_strategy {
            TileMatmulStrategy::Accelerated => hybrid_acc(),
            TileMatmulStrategy::Register => hybrid_reg(),
            TileMatmulStrategy::Auto => match hybrid_acc() {
                Ok(val) => Ok(val),
                Err(_) => hybrid_reg(),
            },
        },
    }
}
