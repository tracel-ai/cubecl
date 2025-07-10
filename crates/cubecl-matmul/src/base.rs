use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{MatmulSetupError, tile::accelerated::AcceleratedMatmul},
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
/// The matmul algorithm to launch
///
/// Most strategies have a selection input that can be overwritten or inferred from minimal information
/// Some strategies must have a specified loading strategy
pub enum Strategy {
    Simple(SyncLoadingStrategy, Selection<SimpleArgs>),
    SimpleBarrier(AsyncLoadingStrategy),
    DoubleBuffering(SyncPartialLoadingStrategy, Selection<DoubleBufferingArgs>),
    SimpleUnit(Selection<SimpleUnitSelectionArgs>),
    DoubleUnit(Selection<DoubleUnitSelectionArgs>),
    OrderedDoubleBuffering(Selection<OrderedSelectionArgs>),
    Naive,
    #[default]
    /// Tries using a Simple matmul, then a SimpleUnit if the former failed
    Auto,
}

#[derive(Debug, Clone)]
/// Which loader to use in simple algorithms
pub enum SyncLoadingStrategy {
    Cyclic,
    Strided,
    Tilewise,
}

#[derive(Debug, Clone)]
/// Which loader to use in double buffering algorithms
pub enum SyncPartialLoadingStrategy {
    Cyclic,
    Tilewise,
    Hybrid,
}

#[derive(Debug, Clone)]
/// Which loader to use in barrier algorithm
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
                        sync_full_strided::SyncFullStridedLoading,
                        sync_full_strided::SyncFullStridedLoading,
                    >,
                >(client, lhs, lhs_scale, rhs, rhs_scale, out, selection)
            }
            SyncLoadingStrategy::Tilewise => layered::launch_ref::<
                R,
                MP,
                SimpleAlgorithm<
                    AcceleratedMatmul,
                    sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                    sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
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
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_cooperative::AsyncFullCooperativeLoading,
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
            AsyncLoadingStrategy::Cyclic => layered::launch_ref::<
                R,
                MP,
                SimpleBarrierAlgorithm<
                    AcceleratedMatmul,
                    async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
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
                    async_full_maximize_slice_length::AsyncFullMaximizeSliceLengthLoading,
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
                    async_full_maximize_unit_count::AsyncFullMaximizeUnitCountLoading,
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
            SyncPartialLoadingStrategy::Cyclic => {
                layered::launch_ref::<R, MP, CyclicDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
                )
            }
            SyncPartialLoadingStrategy::Tilewise => {
                layered::launch_ref::<R, MP, TilewiseDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, lhs_scale, rhs, rhs_scale, out, selection,
                )
            }
            SyncPartialLoadingStrategy::Hybrid => {
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
                    MatmulSetupError::Unavailable(_) => {
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
