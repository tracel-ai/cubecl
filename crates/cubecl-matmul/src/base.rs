use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::tile::accelerated::AcceleratedMatmul,
    kernels::layered::{
        Selection,
        double_buffering::DoubleBufferingArgs,
        double_unit::{DoubleUnitAlgorithm, DoubleUnitSelectionArgs},
        ordered_double_buffering::OrderedSelectionArgs,
        simple::SimpleArgs,
        simple_unit::SimpleUnitSelectionArgs,
    },
};

use super::{
    components::{
        MatmulPrecision,
        global::load::{
            async_full_cooperative, async_full_cyclic, async_full_maximize_slice_length,
            async_full_maximize_unit_count, sync_full_strided, sync_full_tilewise,
        },
        stage::{ColMajorTilingOrder, RowMajorTilingOrder},
    },
    kernels::{
        MatmulSetupError,
        layered::{
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
    },
};

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    Simple(SyncLoadingStrategy, Selection<SimpleArgs>),
    SimpleBarrier(AsyncLoadingStrategy),
    DoubleBuffering(SyncBufferLoadingStrategy, Selection<DoubleBufferingArgs>),
    SimpleUnit(Selection<SimpleUnitSelectionArgs>),
    DoubleUnit(Selection<DoubleUnitSelectionArgs>),
    OrderedDoubleBuffering(Selection<OrderedSelectionArgs>),
    Naive,
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
) -> Result<(), MatmulSetupError> {
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
) -> Result<(), MatmulSetupError> {
    match strategy {
        Strategy::Simple(loading_strategy, selection) => match loading_strategy {
            SyncLoadingStrategy::Cyclic => {
                layered::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(
                    client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
                )
            }
            SyncLoadingStrategy::Strided => {
                layered::launch_ref::<
                    R,
                    MP,
                    SimpleAlgorithm<
                        AcceleratedMatmul,
                        sync_full_strided::LoadingStrategy,
                        sync_full_strided::LoadingStrategy,
                    >,
                >(client, lhs, lhs_scale, rhs, rhs_scale, out, selection)
            }
            SyncLoadingStrategy::Tilewise => layered::launch_ref::<
                R,
                MP,
                SimpleAlgorithm<
                    AcceleratedMatmul,
                    sync_full_tilewise::LoadingStrategy<ColMajorTilingOrder>,
                    sync_full_tilewise::LoadingStrategy<RowMajorTilingOrder>,
                >,
            >(
                client,
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                out,
                &Default::default(),
            ),
        },
        Strategy::SimpleBarrier(loading_strategy) => match loading_strategy {
            AsyncLoadingStrategy::Cooperative => layered::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<AcceleratedMatmul, async_full_cooperative::LoadingStrategy>,
            >(
                client,
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                out,
                &Default::default(),
            ),
            AsyncLoadingStrategy::Cyclic => layered::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
                >,
            >(
                client,
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                out,
                &Default::default(),
            ),
            AsyncLoadingStrategy::MaximizeSliceLength => layered::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_maximize_slice_length::LoadingStrategy,
                >,
            >(
                client,
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                out,
                &Default::default(),
            ),
            AsyncLoadingStrategy::MaximizeUnitCount => layered::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_maximize_unit_count::LoadingStrategy,
                >,
            >(
                client,
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                out,
                &Default::default(),
            ),
            AsyncLoadingStrategy::Tma => {
                layered::matmul_cmma_tma_ref_no_check::<R, MP, SimpleTmaAlgorithm<AcceleratedMatmul>>(
                    client,
                    lhs,
                    lhs_scale,
                    rhs,
                    rhs_scale,
                    out,
                    (false, false),
                    &Default::default(),
                )
            }
        },
        Strategy::DoubleBuffering(loading_strategy, selection) => match loading_strategy {
            SyncBufferLoadingStrategy::Cyclic => {
                layered::launch_ref::<R, MP, CyclicDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
                )
            }
            SyncBufferLoadingStrategy::Tilewise => {
                layered::launch_ref::<R, MP, TilewiseDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
                )
            }
            SyncBufferLoadingStrategy::Hybrid => {
                layered::launch_ref::<R, MP, HybridDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
                )
            }
        },
        Strategy::OrderedDoubleBuffering(selection) => {
            layered::launch_ref::<R, MP, OrderedDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
            )
        }
        Strategy::SimpleUnit(selection) => layered::launch_ref::<R, MP, SimpleUnitAlgorithm>(
            client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
        ),
        Strategy::DoubleUnit(selection) => layered::launch_ref::<R, MP, DoubleUnitAlgorithm>(
            client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
        ),
        Strategy::Naive => {
            // TODO Implement naive with EI and EO
            naive::launch_ref::<R, MP::EI>(client, lhs, rhs, out)?;
            Ok(())
        }
        Strategy::Auto => {
            if let Err(err) = layered::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(
                client,
                lhs,
                lhs_scale,
                rhs,
                rhs_scale,
                out,
                &Default::default(),
            ) {
                match err {
                    super::kernels::MatmulSetupError::Unavailable(_) => {
                        layered::launch_ref::<R, MP, SimpleUnitAlgorithm>(
                            client,
                            lhs,
                            lhs_scale,
                            rhs,
                            rhs_scale,
                            out,
                            &Default::default(),
                        )
                        .unwrap();
                    }
                    _ => panic!("{err:?}"),
                }
            }

            Ok(())
        }
    }
}

pub(crate) fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
