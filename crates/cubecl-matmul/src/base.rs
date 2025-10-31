use cubecl_common::quant::scheme::{QuantScheme, QuantStore, QuantValue};
use cubecl_core::{
    Runtime,
    client::ComputeClient,
    prelude::{CubePrimitive, Numeric, TensorHandleRef},
};

use cubecl_std::tensor::{TensorHandle, into_contiguous_packed, into_contiguous_pitched};
use serde::{Deserialize, Serialize};

use crate::{
    components::{
        AccG, LhsG, MatmulSetupError, RhsG,
        tile::{cmma::CmmaMatmul, io::Filled, mma::MmaMatmul},
    },
    kernels::layered::{
        Selection,
        double_buffering::DoubleBufferingArgs,
        double_unit::{DoubleUnitAlgorithm, DoubleUnitSelectionArgs},
        ordered_double_buffering::OrderedSelectionArgs,
        simple::SimpleArgs,
        simple_unit::SimpleUnitSelectionArgs,
        vecmat::{DoubleVecMatAlgorithm, SimpleVecMatAlgorithm},
    },
};

use super::{
    components::{
        MatmulPrecision,
        global::read::{
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
    Simple {
        read_strategy: SyncReadingStrategy,
        selection: Selection<SimpleArgs>,
        tile_kind: AcceleratedTileKind,
    },
    SimpleBarrier {
        read_strategy: AsyncReadingStrategy,
        tile_kind: AcceleratedTileKind,
    },
    DoubleBuffering {
        read_strategy: SyncPartialReadingStrategy,
        selection: Selection<DoubleBufferingArgs>,
        tile_kind: AcceleratedTileKind,
    },
    SimpleUnit(Selection<SimpleUnitSelectionArgs>),
    DoubleUnit(Selection<DoubleUnitSelectionArgs>),
    SimpleVecMat(Selection<()>),
    DoubleVecMat(Selection<()>),
    OrderedDoubleBuffering {
        selection: Selection<OrderedSelectionArgs>,
        tile_kind: AcceleratedTileKind,
    },
    Naive,
    #[default]
    /// Tries using a Simple matmul, then a SimpleUnit if the former failed
    Auto,
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in simple algorithms
pub enum SyncReadingStrategy {
    Cyclic,
    Strided,
    Tilewise,
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in double buffering algorithms
pub enum SyncPartialReadingStrategy {
    Cyclic,
    Tilewise,
    Hybrid,
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in barrier algorithm
pub enum AsyncReadingStrategy {
    Cooperative,
    Cyclic,
    MaximizeSliceLength,
    MaximizeUnitCount,
    Tma,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
/// Which tile matmul to use for accelerated algorithms
pub enum AcceleratedTileKind {
    #[default]
    Cmma,
    Mma,
}

macro_rules! with_tile_kind {
    ($kind: expr, $T: ident, $launch: expr) => {
        match $kind {
            AcceleratedTileKind::Cmma => {
                type $T = CmmaMatmul<Filled>;
                ($launch)()
            }
            AcceleratedTileKind::Mma => {
                type $T = MmaMatmul<Filled>;
                ($launch)()
            }
        }
    };
}

pub enum MatmulInputHandle<R: Runtime, E: CubePrimitive, S: CubePrimitive = f32> {
    Normal(TensorHandle<R, E>),
    Quantized {
        data: TensorHandle<R, E>,
        scale: TensorHandle<R, S>,
        shape: Vec<usize>,
        scheme: QuantScheme,
    },
}

impl<R: Runtime, E: Numeric> MatmulInputHandle<R, E> {
    pub fn as_ref(&self) -> MatmulInputHandleRef<'_, R> {
        match self {
            MatmulInputHandle::Normal(handle) => MatmulInputHandleRef::Normal(handle.as_ref()),
            MatmulInputHandle::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => MatmulInputHandleRef::Quantized {
                data: data.as_ref(),
                scale: scale.as_ref(),
                shape,
                scheme,
            },
        }
    }

    pub fn from_ref(handle: &MatmulInputHandleRef<'_, R>) -> Self {
        match handle {
            MatmulInputHandleRef::Normal(handle) => {
                MatmulInputHandle::Normal(TensorHandle::from_ref(handle))
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => MatmulInputHandle::Quantized {
                data: TensorHandle::from_ref(data),
                scale: TensorHandle::from_ref(scale),
                shape: shape.to_vec(),
                scheme: **scheme,
            },
        }
    }

    pub fn data(&self) -> &TensorHandle<R, E> {
        match self {
            MatmulInputHandle::Normal(handle) => handle,
            MatmulInputHandle::Quantized { data, .. } => data,
        }
    }

    pub fn swap_dims(&mut self, dim0: usize, dim1: usize) {
        match self {
            MatmulInputHandle::Normal(handle) => {
                handle.shape.swap(dim0, dim1);
                handle.strides.swap(dim0, dim1);
            }
            MatmulInputHandle::Quantized {
                data, scale, shape, ..
            } => {
                data.shape.swap(dim0, dim1);
                data.strides.swap(dim0, dim1);
                if scale.shape.len() == data.shape.len() {
                    scale.shape.swap(dim0, dim1);
                    scale.strides.swap(dim0, dim1);
                }
                shape.swap(dim0, dim1);
            }
        }
    }
}

impl<R: Runtime, E: CubePrimitive> Clone for MatmulInputHandle<R, E> {
    fn clone(&self) -> Self {
        match self {
            Self::Normal(handle) => Self::Normal(handle.clone()),
            Self::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => Self::Quantized {
                data: data.clone(),
                scale: scale.clone(),
                shape: shape.clone(),
                scheme: *scheme,
            },
        }
    }
}

#[derive(Debug)]
pub enum MatmulInputHandleRef<'a, R: Runtime> {
    Normal(TensorHandleRef<'a, R>),
    Quantized {
        data: TensorHandleRef<'a, R>,
        scale: TensorHandleRef<'a, R>,
        /// Unpacked shape, excluding padding
        shape: &'a [usize],
        scheme: &'a QuantScheme,
    },
}

impl<'a, R: Runtime> Clone for MatmulInputHandleRef<'a, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: Runtime> Copy for MatmulInputHandleRef<'a, R> {}

impl<'a, R: Runtime> MatmulInputHandleRef<'a, R> {
    pub fn new(data: TensorHandleRef<'a, R>) -> Self {
        Self::Normal(data)
    }

    pub fn quantized(
        data: TensorHandleRef<'a, R>,
        scale: TensorHandleRef<'a, R>,
        shape: &'a [usize],
        scheme: &'a QuantScheme,
    ) -> Self {
        Self::Quantized {
            data,
            scale,
            shape,
            scheme,
        }
    }

    pub fn data(&self) -> &TensorHandleRef<'a, R> {
        match self {
            MatmulInputHandleRef::Normal(handle) => handle,
            MatmulInputHandleRef::Quantized { data, .. } => data,
        }
    }

    pub fn data_mut(&mut self) -> &mut TensorHandleRef<'a, R> {
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

    pub fn scheme(&self) -> Option<&QuantScheme> {
        match self {
            MatmulInputHandleRef::Normal(_) => None,
            MatmulInputHandleRef::Quantized { scheme, .. } => Some(scheme),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            MatmulInputHandleRef::Normal(handle) => handle.shape,
            MatmulInputHandleRef::Quantized { shape, .. } => shape,
        }
    }

    pub fn into_contiguous<E: Numeric>(
        &self,
        client: &ComputeClient<R::Server>,
    ) -> MatmulInputHandle<R, E> {
        match self {
            MatmulInputHandleRef::Normal(data) => {
                MatmulInputHandle::Normal(into_contiguous_pitched::<R, E>(client, data))
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => {
                let data = match scheme.store {
                    // e2m1 has native packing (e2m1x2) so also needs to be re-packed
                    QuantStore::Native if scheme.value == QuantValue::E2M1 => {
                        let data = into_contiguous_packed::<R, u8>(client, data, shape, 2);
                        // Unsafely cast to E
                        TensorHandle::from_ref(&data.as_ref())
                    }
                    QuantStore::U32 => {
                        let data = into_contiguous_packed::<R, u32>(
                            client,
                            data,
                            shape,
                            scheme.num_quants() as u32,
                        );
                        // Unsafely cast to E
                        TensorHandle::from_ref(&data.as_ref())
                    }
                    _ => into_contiguous_pitched::<R, E>(client, data),
                };
                MatmulInputHandle::Quantized {
                    data,
                    scale: TensorHandle::from_ref(scale),
                    shape: shape.to_vec(),
                    scheme: **scheme,
                }
            }
        }
    }
}

#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, MP: MatmulPrecision>(
    strategy: &Strategy,
    client: &ComputeClient<R::Server>,
    lhs: MatmulInputHandle<R, LhsG<MP>>,
    rhs: MatmulInputHandle<R, RhsG<MP>>,
    out: TensorHandle<R, AccG<MP>>,
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
    client: &ComputeClient<R::Server>,
    lhs: &MatmulInputHandleRef<R>,
    rhs: &MatmulInputHandleRef<R>,
    out: &TensorHandleRef<R>,
) -> Result<(), MatmulSetupError> {
    match strategy {
        Strategy::Simple {
            read_strategy,
            selection,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            SyncReadingStrategy::Cyclic => {
                layered::launch_ref::<R, MP, SimpleAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection,
                )
            }
            SyncReadingStrategy::Strided => layered::launch_ref::<
                R,
                MP,
                SimpleAlgorithm<
                    Accelerated,
                    sync_full_strided::SyncFullStridedLoading,
                    sync_full_strided::SyncFullStridedLoading,
                >,
            >(client, lhs, rhs, out, selection),
            SyncReadingStrategy::Tilewise => {
                layered::launch_ref::<
                    R,
                    MP,
                    SimpleAlgorithm<
                        Accelerated,
                        sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                        sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
                    >,
                >(client, lhs, rhs, out, &Default::default())
            }
        }),
        Strategy::SimpleBarrier {
            read_strategy,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            AsyncReadingStrategy::Cooperative => {
                layered::launch_ref::<
                    R,
                    MP,
                    SimpleBarrierAlgorithm<
                        Accelerated,
                        async_full_cooperative::AsyncFullCooperativeLoading,
                    >,
                >(client, lhs, rhs, out, &Default::default())
            }
            AsyncReadingStrategy::Cyclic => {
                layered::launch_ref::<
                    R,
                    MP,
                    SimpleBarrierAlgorithm<
                        Accelerated,
                        async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                    >,
                >(client, lhs, rhs, out, &Default::default())
            }
            AsyncReadingStrategy::MaximizeSliceLength => {
                layered::launch_ref::<
                    R,
                    MP,
                    SimpleBarrierAlgorithm<
                        Accelerated,
                        async_full_maximize_slice_length::AsyncFullMaximizeSliceLengthLoading,
                    >,
                >(client, lhs, rhs, out, &Default::default())
            }
            AsyncReadingStrategy::MaximizeUnitCount => {
                layered::launch_ref::<
                    R,
                    MP,
                    SimpleBarrierAlgorithm<
                        Accelerated,
                        async_full_maximize_unit_count::AsyncFullMaximizeUnitCountLoading,
                    >,
                >(client, lhs, rhs, out, &Default::default())
            }
            AsyncReadingStrategy::Tma => {
                layered::matmul_cmma_tma_ref_no_check::<R, MP, SimpleTmaAlgorithm<Accelerated>>(
                    client,
                    lhs,
                    rhs,
                    out,
                    (false, false),
                    &Default::default(),
                )
            }
        }),
        Strategy::DoubleBuffering {
            read_strategy,
            selection,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            SyncPartialReadingStrategy::Cyclic => {
                layered::launch_ref::<R, MP, CyclicDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection,
                )
            }
            SyncPartialReadingStrategy::Tilewise => {
                layered::launch_ref::<R, MP, TilewiseDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection,
                )
            }
            SyncPartialReadingStrategy::Hybrid => {
                layered::launch_ref::<R, MP, HybridDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection,
                )
            }
        }),
        Strategy::OrderedDoubleBuffering {
            selection,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || layered::launch_ref::<
            R,
            MP,
            OrderedDoubleBufferingAlgorithm<Accelerated>,
        >(
            client, lhs, rhs, out, selection,
        )),
        Strategy::SimpleUnit(selection) => {
            layered::launch_ref::<R, MP, SimpleUnitAlgorithm>(client, lhs, rhs, out, selection)
        }
        Strategy::DoubleUnit(selection) => {
            layered::launch_ref::<R, MP, DoubleUnitAlgorithm>(client, lhs, rhs, out, selection)
        }
        Strategy::Naive => {
            naive::launch_ref::<R, LhsG<MP>, AccG<MP>>(client, lhs, rhs, out)?;
            Ok(())
        }
        Strategy::Auto => {
            if let Err(err) = layered::launch_ref::<R, MP, SimpleAlgorithm<CmmaMatmul<Filled>>>(
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
        Strategy::SimpleVecMat(selection) => {
            layered::launch_ref::<R, MP, SimpleVecMatAlgorithm>(client, lhs, rhs, out, selection)
        }
        Strategy::DoubleVecMat(selection) => {
            layered::launch_ref::<R, MP, DoubleVecMatAlgorithm>(client, lhs, rhs, out, selection)
        }
    }
}
