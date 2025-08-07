use cubecl_core::{
    Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{LhsG, MatmulSetupError, RhsG, tile::accelerated::AcceleratedMatmul},
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

pub enum MatmulInputHandle<R: Runtime, E: Numeric> {
    Normal(TensorHandle<R, E>),
    Quantized {
        data: TensorHandle<R, E>,
        scale: TensorHandle<R, f32>,
    },
}

impl<R: Runtime, E: Numeric> MatmulInputHandle<R, E> {
    pub fn as_ref(&self) -> MatmulInputHandleRef<'_, R> {
        match self {
            MatmulInputHandle::Normal(handle) => MatmulInputHandleRef::Normal(handle.as_ref()),
            MatmulInputHandle::Quantized { data, scale } => MatmulInputHandleRef::Quantized {
                data: data.as_ref(),
                scale: scale.as_ref(),
            },
        }
    }
}

impl<R: Runtime, E: Numeric> Clone for MatmulInputHandle<R, E> {
    fn clone(&self) -> Self {
        match self {
            Self::Normal(handle) => Self::Normal(handle.clone()),
            Self::Quantized { data, scale } => Self::Quantized {
                data: data.clone(),
                scale: scale.clone(),
            },
        }
    }
}

pub enum MatmulInputHandleRef<'a, R: Runtime> {
    Normal(TensorHandleRef<'a, R>),
    Quantized {
        data: TensorHandleRef<'a, R>,
        scale: TensorHandleRef<'a, R>,
    },
}

impl<'a, R: Runtime> MatmulInputHandleRef<'a, R> {
    pub fn data(&self) -> &TensorHandleRef<'a, R> {
        match self {
            MatmulInputHandleRef::Normal(handle) => handle,
            MatmulInputHandleRef::Quantized { data, .. } => data,
        }
    }

    pub fn scale(&self) -> Option<&TensorHandleRef<'a, R>> {
        match self {
            MatmulInputHandleRef::Normal(_) => None,
            MatmulInputHandleRef::Quantized { scale, .. } => Some(scale),
        }
    }
}

#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, MP: MatmulPrecision>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: MatmulInputHandle<R, LhsG<MP>>,
    rhs: MatmulInputHandle<R, RhsG<MP>>,
    out: TensorHandle<R, MP::EO>,
) -> Result<(), MatmulSetupError> {
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
    lhs: &MatmulInputHandleRef<R>,
    rhs: &MatmulInputHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulSetupError> {
    match strategy {
        Strategy::Simple(loading_strategy, selection) => match loading_strategy {
            SyncLoadingStrategy::Cyclic => {
                layered::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(
                    client, lhs, rhs, out, selection,
                )
            }
            SyncLoadingStrategy::Strided => layered::launch_ref::<
                R,
                MP,
                SimpleAlgorithm<
                    AcceleratedMatmul,
                    sync_full_strided::SyncFullStridedLoading,
                    sync_full_strided::SyncFullStridedLoading,
                >,
            >(client, lhs, rhs, out, selection),
            SyncLoadingStrategy::Tilewise => {
                layered::launch_ref::<
                    R,
                    MP,
                    SimpleAlgorithm<
                        AcceleratedMatmul,
                        sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                        sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
                    >,
                >(client, lhs, rhs, out, &Default::default())
            }
        },
        Strategy::SimpleBarrier(loading_strategy) => {
            match loading_strategy {
                AsyncLoadingStrategy::Cooperative => {
                    layered::launch_ref::<
                        R,
                        MP,
                        SimpleBarrierAlgorithm<
                            AcceleratedMatmul,
                            async_full_cooperative::AsyncFullCooperativeLoading,
                        >,
                    >(client, lhs, rhs, out, &Default::default())
                }
                AsyncLoadingStrategy::Cyclic => {
                    layered::launch_ref::<
                        R,
                        MP,
                        SimpleBarrierAlgorithm<
                            AcceleratedMatmul,
                            async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                        >,
                    >(client, lhs, rhs, out, &Default::default())
                }
                AsyncLoadingStrategy::MaximizeSliceLength => {
                    layered::launch_ref::<
                        R,
                        MP,
                        SimpleBarrierAlgorithm<
                            AcceleratedMatmul,
                            async_full_maximize_slice_length::AsyncFullMaximizeSliceLengthLoading,
                        >,
                    >(client, lhs, rhs, out, &Default::default())
                }
                AsyncLoadingStrategy::MaximizeUnitCount => {
                    layered::launch_ref::<
                        R,
                        MP,
                        SimpleBarrierAlgorithm<
                            AcceleratedMatmul,
                            async_full_maximize_unit_count::AsyncFullMaximizeUnitCountLoading,
                        >,
                    >(client, lhs, rhs, out, &Default::default())
                }
                AsyncLoadingStrategy::Tma => {
                    layered::matmul_cmma_tma_ref_no_check::<
                        R,
                        MP,
                        SimpleTmaAlgorithm<AcceleratedMatmul>,
                    >(client, lhs, rhs, out, (false, false), &Default::default())
                }
            }
        }
        Strategy::DoubleBuffering(loading_strategy, selection) => match loading_strategy {
            SyncPartialLoadingStrategy::Cyclic => {
                layered::launch_ref::<R, MP, CyclicDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, rhs, out, selection,
                )
            }
            SyncPartialLoadingStrategy::Tilewise => {
                layered::launch_ref::<R, MP, TilewiseDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, rhs, out, selection,
                )
            }
            SyncPartialLoadingStrategy::Hybrid => {
                layered::launch_ref::<R, MP, HybridDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                    client, lhs, rhs, out, selection,
                )
            }
        },
        Strategy::OrderedDoubleBuffering(selection) => {
            layered::launch_ref::<R, MP, OrderedDoubleBufferingAlgorithm<AcceleratedMatmul>>(
                client, lhs, rhs, out, selection,
            )
        }
        Strategy::SimpleUnit(selection) => {
            layered::launch_ref::<R, MP, SimpleUnitAlgorithm>(client, lhs, rhs, out, selection)
        }
        Strategy::DoubleUnit(selection) => {
            layered::launch_ref::<R, MP, DoubleUnitAlgorithm>(client, lhs, rhs, out, selection)
        }
        Strategy::Naive => {
            // Warning: this assumes Lhs, Rhs and Output have the same type
            naive::launch_ref::<R, LhsG<MP>>(client, lhs.data(), rhs.data(), out)?;
            Ok(())
        }
        Strategy::Auto => {
            if let Err(err) = layered::launch_ref::<R, MP, SimpleAlgorithm<AcceleratedMatmul>>(
                client,
                lhs,
                rhs,
                out,
                &Default::default(),
            ) {
                match err {
                    MatmulSetupError::Unavailable(_) => {
                        layered::launch_ref::<R, MP, SimpleUnitAlgorithm>(
                            client,
                            lhs,
                            rhs,
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
