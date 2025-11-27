use std::fmt::Display;

use cubecl_common::quant::scheme::{QuantScheme, QuantStore, QuantValue};
use cubecl_core::{
    Runtime,
    client::ComputeClient,
    ir::StorageType,
    prelude::{CubePrimitive, TensorHandleRef},
    server::LaunchError,
};

use cubecl_std::tensor::{TensorHandle, into_contiguous_packed, into_contiguous_pitched};
use serde::{Deserialize, Serialize};

use crate::{
    components::{
        MatmulElems, MatmulSetupError,
        global::read::{
            async_partial_cyclic::AsyncPartialCyclicLoading,
            async_partial_strided::AsyncPartialStridedLoading,
        },
        tile::{cmma::CmmaMatmul, io::Filled, mma::MmaMatmul},
    },
    kernels::layered::{
        Selection,
        double_buffering::*,
        double_unit::{DoubleUnitAlgorithm, DoubleUnitSelectionArgs},
        ordered_double_buffering::OrderedSelectionArgs,
        simple::SimpleArgs,
        simple_unit::SimpleUnitSelectionArgs,
        specialized::SpecializedAlgorithm,
        vecmat::{DoubleVecMatAlgorithm, SimpleVecMatAlgorithm},
    },
};

use super::{
    components::{
        global::read::{
            async_full_cooperative, async_full_cyclic, sync_full_strided, sync_full_tilewise,
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
            simple::{SimpleAlgorithm, SimpleTmaAlgorithm},
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
        read_strategy: ReadingStrategy,
        selection: Selection<SimpleArgs>,
        tile_kind: AcceleratedTileKind,
    },
    DoubleBuffering {
        read_strategy: PartialReadingStrategy,
        selection: Selection<DoubleBufferingArgs>,
        tile_kind: AcceleratedTileKind,
    },
    Specialized {
        read_strategy: AsyncPartialReadingStrategy,
        selection: Selection<()>,
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

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Simple {
                read_strategy,
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!("matmul_simple_{read_strategy}_{tile_kind}"))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        if args.multi_rows {
                            f.write_str("_multirows")?;
                        }
                    }
                };
            }
            Strategy::DoubleBuffering {
                read_strategy,
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!(
                    "matmul_double_buffering_{read_strategy}_{tile_kind}"
                ))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        if args.specialized {
                            f.write_str("_specialized")?;
                        }
                    }
                };
            }
            Strategy::Specialized {
                read_strategy,
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!(
                    "matmul_specialized_{read_strategy}_{tile_kind}"
                ))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(_) => {}
                };
            }
            Strategy::SimpleUnit(selection) => {
                f.write_fmt(format_args!("matmul_simple_unit"))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        f.write_fmt(format_args!("_{}", args.tile_size))?;
                    }
                };
            }
            Strategy::DoubleUnit(selection) => {
                f.write_str("matmul_double_buffering_unit")?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        f.write_fmt(format_args!("_{}", args.tile_size))?;
                    }
                };
            }
            Strategy::SimpleVecMat(selection) => {
                f.write_str("vecmat_simple")?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(_) => {}
                };
            }
            Strategy::DoubleVecMat(selection) => {
                f.write_str("vecmat_double_buffering")?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(_) => {}
                };
            }
            Strategy::OrderedDoubleBuffering {
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!("matmul_double_buffering_ordered_{tile_kind}"))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        if let Some(k) = args.partition_k {
                            f.write_fmt(format_args!("_partition_k{}", k))?;
                        }
                        if let Some(r) = args.row_count {
                            f.write_fmt(format_args!("_row_count{}", r))?;
                        }
                        if let Some(r) = args.rows_per_plane {
                            f.write_fmt(format_args!("_row_per_plane{}", r))?;
                        }
                    }
                };
            }
            Strategy::Naive => f.write_str("matmul_naive")?,
            Strategy::Auto => f.write_str("matmul_auto")?,
        };

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in simple algorithms
pub enum ReadingStrategy {
    Cyclic,
    Strided,
    Tilewise,
    AsyncCooperative,
    AsyncCyclic,
    Tma,
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in double buffering algorithms
pub enum PartialReadingStrategy {
    Cyclic,
    Tilewise,
    Hybrid,
    Tma,
    AsyncCyclic,
    AsyncStrided,
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in specialized algorithms
pub enum AsyncPartialReadingStrategy {
    Cyclic,
    Strided,
    Tma,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Which tile matmul to use for accelerated algorithms
pub enum AcceleratedTileKind {
    #[default]
    Cmma,
    Mma,
}

// Display implementations are used to combine and save names when autotuning.

impl Display for AcceleratedTileKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceleratedTileKind::Cmma => f.write_str("cmma"),
            AcceleratedTileKind::Mma => f.write_str("mma"),
        }
    }
}

impl Display for ReadingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadingStrategy::Cyclic => f.write_str("cyclic"),
            ReadingStrategy::Strided => f.write_str("strided"),
            ReadingStrategy::Tilewise => f.write_str("tilewise"),
            ReadingStrategy::AsyncCooperative => f.write_str("async_cooperative"),
            ReadingStrategy::AsyncCyclic => f.write_str("async_cyclic"),
            ReadingStrategy::Tma => f.write_str("tma"),
        }
    }
}

impl Display for PartialReadingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartialReadingStrategy::Cyclic => f.write_str("cyclic"),
            PartialReadingStrategy::Tilewise => f.write_str("tilewise"),
            PartialReadingStrategy::Hybrid => f.write_str("hybrid"),
            PartialReadingStrategy::Tma => f.write_str("tma"),
            PartialReadingStrategy::AsyncCyclic => f.write_str("async_cyclic"),
            PartialReadingStrategy::AsyncStrided => f.write_str("async_strided"),
        }
    }
}

impl Display for AsyncPartialReadingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsyncPartialReadingStrategy::Cyclic => f.write_str("cyclic"),
            AsyncPartialReadingStrategy::Strided => f.write_str("strided"),
            AsyncPartialReadingStrategy::Tma => f.write_str("tma"),
        }
    }
}

macro_rules! with_tile_kind {
    ($kind: expr, $T: ident, $launch: expr) => {
        match $kind {
            AcceleratedTileKind::Cmma => {
                type $T = CmmaMatmul<Filled>;
                ($launch)()
            }
            AcceleratedTileKind::Mma => {
                type $T = MmaMatmul;
                ($launch)()
            }
        }
    };
}

pub enum MatmulInputHandle<R: Runtime> {
    Normal(TensorHandle<R>),
    Quantized {
        data: TensorHandle<R>,
        scale: TensorHandle<R>,
        shape: Vec<usize>,
        scheme: QuantScheme,
    },
}

impl<R: Runtime> MatmulInputHandle<R> {
    pub fn as_ref(&self) -> MatmulInputHandleRef<'_, R> {
        match self {
            MatmulInputHandle::Normal(handle) => {
                MatmulInputHandleRef::Normal(handle.as_ref(), handle.dtype)
            }
            MatmulInputHandle::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => MatmulInputHandleRef::Quantized {
                data: data.as_ref(),
                scale: scale.as_ref(),
                data_dtype: data.dtype,
                scale_dtype: scale.dtype,
                shape,
                scheme,
            },
        }
    }

    pub fn from_ref(handle: &MatmulInputHandleRef<'_, R>) -> Self {
        match handle {
            MatmulInputHandleRef::Normal(handle, dtype) => {
                MatmulInputHandle::Normal(TensorHandle::from_ref(handle, *dtype))
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
                data_dtype,
                scale_dtype,
            } => MatmulInputHandle::Quantized {
                data: TensorHandle::from_ref(data, *data_dtype),
                scale: TensorHandle::from_ref(scale, *scale_dtype),
                shape: shape.to_vec(),
                scheme: **scheme,
            },
        }
    }

    pub fn data(&self) -> &TensorHandle<R> {
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

impl<R: Runtime> Clone for MatmulInputHandle<R> {
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
    Normal(TensorHandleRef<'a, R>, StorageType),
    Quantized {
        data: TensorHandleRef<'a, R>,
        data_dtype: StorageType,
        scale: TensorHandleRef<'a, R>,
        scale_dtype: StorageType,
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
    pub fn new(data: TensorHandleRef<'a, R>, dtype: StorageType) -> Self {
        Self::Normal(data, dtype)
    }

    pub fn quantized(
        data: TensorHandleRef<'a, R>,
        scale: TensorHandleRef<'a, R>,
        shape: &'a [usize],
        scheme: &'a QuantScheme,
        data_dtype: StorageType,
        scale_dtype: StorageType,
    ) -> Self {
        Self::Quantized {
            data,
            scale,
            shape,
            scheme,
            data_dtype,
            scale_dtype,
        }
    }

    pub fn data(&self) -> &TensorHandleRef<'a, R> {
        match self {
            MatmulInputHandleRef::Normal(handle, ..) => handle,
            MatmulInputHandleRef::Quantized { data, .. } => data,
        }
    }

    pub fn data_mut(&mut self) -> &mut TensorHandleRef<'a, R> {
        match self {
            MatmulInputHandleRef::Normal(handle, ..) => handle,
            MatmulInputHandleRef::Quantized { data, .. } => data,
        }
    }

    pub fn scale(&self) -> Option<&TensorHandleRef<'a, R>> {
        match self {
            MatmulInputHandleRef::Normal(..) => None,
            MatmulInputHandleRef::Quantized { scale, .. } => Some(scale),
        }
    }

    pub fn scheme(&self) -> Option<&QuantScheme> {
        match self {
            MatmulInputHandleRef::Normal(..) => None,
            MatmulInputHandleRef::Quantized { scheme, .. } => Some(scheme),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            MatmulInputHandleRef::Normal(handle, ..) => handle.shape,
            MatmulInputHandleRef::Quantized { shape, .. } => shape,
        }
    }

    pub fn into_contiguous(
        &self,
        client: &ComputeClient<R>,
    ) -> Result<MatmulInputHandle<R>, LaunchError> {
        let val = match self {
            MatmulInputHandleRef::Normal(data, dtype) => {
                MatmulInputHandle::Normal(into_contiguous_pitched(client, data, *dtype)?)
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
                data_dtype,
                scale_dtype,
            } => {
                let data = match scheme.store {
                    // e2m1 has native packing (e2m1x2) so also needs to be re-packed
                    QuantStore::Native if scheme.value == QuantValue::E2M1 => {
                        let data = into_contiguous_packed(
                            client,
                            data,
                            shape,
                            2,
                            u8::as_type_native_unchecked(),
                        )?;
                        // Unsafely cast to E
                        TensorHandle::from_ref(&data.as_ref(), *data_dtype)
                    }
                    QuantStore::U32 => {
                        let data = into_contiguous_packed(
                            client,
                            data,
                            shape,
                            scheme.num_quants() as u32,
                            u32::as_type_native_unchecked(),
                        )?;
                        // Unsafely cast to E
                        TensorHandle::from_ref(&data.as_ref(), *data_dtype)
                    }
                    _ => into_contiguous_pitched(client, data, *data_dtype)?,
                };
                MatmulInputHandle::Quantized {
                    data,
                    scale: TensorHandle::from_ref(scale, *scale_dtype),
                    shape: shape.to_vec(),
                    scheme: **scheme,
                }
            }
        };

        Ok(val)
    }
}

#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: MatmulInputHandle<R>,
    rhs: MatmulInputHandle<R>,
    out: TensorHandle<R>,
    mut dtypes: MatmulElems,
) -> Result<(), MatmulSetupError> {
    launch_ref(
        strategy,
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
        &mut dtypes,
    )
}

#[allow(clippy::result_large_err)]
/// Launches a matrix multiplication kernel..
///
/// # Notes
///
/// The matmul elements may get changed during selection for improved performance when
/// the hardware supports it.
/// Only the inner element types may change such as the stage or register element types.
pub fn launch_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<R>,
    rhs: &MatmulInputHandleRef<R>,
    out: &TensorHandleRef<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    match strategy {
        Strategy::Simple {
            read_strategy,
            selection,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            ReadingStrategy::Cyclic => {
                layered::launch_ref::<R, SimpleAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes,
                )
            }
            ReadingStrategy::Strided => layered::launch_ref::<
                R,
                SimpleAlgorithm<
                    Accelerated,
                    sync_full_strided::SyncFullStridedLoading,
                    sync_full_strided::SyncFullStridedLoading,
                >,
            >(client, lhs, rhs, out, selection, dtypes),
            ReadingStrategy::Tilewise => {
                layered::launch_ref::<
                    R,
                    SimpleAlgorithm<
                        Accelerated,
                        sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                        sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
                    >,
                >(client, lhs, rhs, out, selection, dtypes)
            }
            ReadingStrategy::AsyncCooperative => {
                layered::launch_ref::<
                    R,
                    SimpleAlgorithm<
                        Accelerated,
                        async_full_cooperative::AsyncFullCooperativeLoading,
                        async_full_cooperative::AsyncFullCooperativeLoading,
                    >,
                >(client, lhs, rhs, out, selection, dtypes)
            }
            ReadingStrategy::AsyncCyclic => {
                layered::launch_ref::<
                    R,
                    SimpleAlgorithm<
                        Accelerated,
                        async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                        async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
                    >,
                >(client, lhs, rhs, out, selection, dtypes)
            }
            ReadingStrategy::Tma => layered::launch_ref_tma::<R, SimpleTmaAlgorithm<Accelerated>>(
                client, lhs, rhs, out, selection, dtypes
            ),
        }),
        Strategy::DoubleBuffering {
            read_strategy,
            selection,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            PartialReadingStrategy::Cyclic => {
                layered::launch_ref::<R, CyclicDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes,
                )
            }
            PartialReadingStrategy::Tilewise => {
                layered::launch_ref::<R, TilewiseDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes,
                )
            }
            PartialReadingStrategy::Hybrid => {
                layered::launch_ref::<R, HybridDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes,
                )
            }
            PartialReadingStrategy::Tma => {
                layered::launch_ref_tma::<R, TmaDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes,
                )
            }
            PartialReadingStrategy::AsyncCyclic => {
                layered::launch_ref::<R, AsyncCyclicDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes,
                )
            }
            PartialReadingStrategy::AsyncStrided => {
                layered::launch_ref::<R, AsyncStridedDoubleBufferingAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes,
                )
            }
        }),
        Strategy::Specialized {
            read_strategy,
            selection,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || match read_strategy {
            AsyncPartialReadingStrategy::Cyclic => layered::launch_ref::<
                R,
                SpecializedAlgorithm<Accelerated, AsyncPartialCyclicLoading<ColMajorTilingOrder>>,
            >(
                client, lhs, rhs, out, selection, dtypes
            ),
            AsyncPartialReadingStrategy::Strided =>
                layered::launch_ref::<
                    R,
                    SpecializedAlgorithm<Accelerated, AsyncPartialStridedLoading>,
                >(client, lhs, rhs, out, selection, dtypes),
            AsyncPartialReadingStrategy::Tma =>
                layered::launch_ref_tma::<R, SpecializedAlgorithm<Accelerated>>(
                    client, lhs, rhs, out, selection, dtypes
                ),
        }),
        Strategy::OrderedDoubleBuffering {
            selection,
            tile_kind,
        } => with_tile_kind!(tile_kind, Accelerated, || layered::launch_ref::<
            R,
            OrderedDoubleBufferingAlgorithm<Accelerated>,
        >(
            client, lhs, rhs, out, selection, dtypes
        )),
        Strategy::SimpleUnit(selection) => {
            layered::launch_ref::<R, SimpleUnitAlgorithm>(client, lhs, rhs, out, selection, dtypes)
        }
        Strategy::DoubleUnit(selection) => {
            layered::launch_ref::<R, DoubleUnitAlgorithm>(client, lhs, rhs, out, selection, dtypes)
        }
        Strategy::Naive => {
            naive::launch_ref(client, lhs, rhs, out, dtypes)?;
            Ok(())
        }
        Strategy::Auto => {
            if let Err(err) = layered::launch_ref::<R, SimpleAlgorithm<CmmaMatmul<Filled>>>(
                client,
                lhs,
                rhs,
                out,
                &Default::default(),
                dtypes,
            ) {
                match err {
                    MatmulSetupError::Unavailable(_) => {
                        layered::launch_ref::<R, SimpleUnitAlgorithm>(
                            client,
                            lhs,
                            rhs,
                            out,
                            &Default::default(),
                            dtypes,
                        )
                        .unwrap();
                    }
                    _ => panic!("{err:?}"),
                }
            }

            Ok(())
        }
        Strategy::SimpleVecMat(selection) => layered::launch_ref::<R, SimpleVecMatAlgorithm>(
            client, lhs, rhs, out, selection, dtypes,
        ),
        Strategy::DoubleVecMat(selection) => layered::launch_ref::<R, DoubleVecMatAlgorithm>(
            client, lhs, rhs, out, selection, dtypes,
        ),
    }
}
