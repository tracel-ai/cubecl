use std::any::TypeId;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, server::TensorMapMeta, unexpanded};
use cubecl_std::{
    CubeOption, CubeOptionArgs, CubeOptionExpand, FastDivmod,
    tensor::{
        View,
        launch::ViewArg,
        layout::{Coords1d, Coords2d, Coords3d},
    },
};

use crate::{
    MatmulInputHandleRef,
    components::{
        self, MatmulIdent, MatmulLineSizes, MatmulProblem, MatmulSelection,
        batch::{BatchConfig, SliceIndex},
        global::{
            GlobalConfig,
            memory::{
                BatchedGlobalLayout, BatchedGlobalLayoutLaunch, BatchedGlobalScaleLayout,
                CachedBatchedGlobalLayout, GlobalLayoutConfig, SimpleTmaGlobalLayout,
                SimpleTmaGlobalLayoutLaunch,
            },
        },
    },
};

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        config: impl BatchConfig,
    ) -> Self::RuntimeArg<'a, R>;
}

/// Create the output runtime argument for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        out: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        config: impl BatchConfig,
    ) -> Self::RuntimeArg<'a, R>;
}

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<EO: Numeric>: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric>: CubeType;

    /// Init the state.
    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric, G: GlobalConfig>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        #[comptime] config: G,
    ) -> Self::State<Lhs, Rhs, EO>;

    fn get_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> BatchedMatrix<Lhs> {
        unexpanded!()
    }
    fn get_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> BatchedMatrix<Rhs> {
        unexpanded!()
    }
    fn get_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<BatchedMatrix<EO>> {
        unexpanded!()
    }
    fn get_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        unexpanded!()
    }
}

#[derive(Clone, Copy)]
/// Identification of the [tensor input](TensorInput).
pub enum TensorInputIdent {
    Lhs,
    Rhs,
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorArgs;

#[derive(CubeLaunch, CubeType)]
pub enum BatchedMatrix<I: Numeric> {
    Strided(StridedBatchedMatrix<I>),
    Viewed(View<Line<I>, Coords3d>),
}

#[derive(CubeLaunch, CubeType)]
pub struct StridedBatchedMatrix<I: Numeric> {
    batch_shape: Sequence<FastDivmod>,
    batch_strides: Sequence<u32>,
    shape_row: u32,
    shape_col: u32,
    stride_row: u32,
    stride_col: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    packing: u32,
    buffer: View<Line<I>, Coords1d>,
    #[cube(comptime)]
    config: GlobalLayoutConfig,
}

#[cube]
impl<I: Numeric> BatchedMatrix<I> {
    pub fn cloned(&self) -> Self {
        #[allow(unused_variables)]
        match self {
            BatchedMatrix::Strided(tensor) => {
                intrinsic!(|scope| { BatchedMatrixExpand::Strided(tensor.clone()) })
            }
            BatchedMatrix::Viewed(view) => {
                intrinsic!(|scope| { BatchedMatrixExpand::Viewed(view.clone()) })
            }
        }
    }
    pub fn shape_col(&self) -> u32 {
        match self {
            BatchedMatrix::Strided(tensor) => tensor.shape_col,
            BatchedMatrix::Viewed(view) => view.shape().2,
        }
    }
    pub fn into_matrix(self, batch_nth: u32) -> View<Line<I>, Coords2d> {
        match self {
            BatchedMatrix::Strided(tensor) => {
                let layout = CachedBatchedGlobalLayout::new(
                    tensor.batch_strides,
                    tensor.batch_shape,
                    tensor.shape_row,
                    tensor.shape_col,
                    tensor.stride_row,
                    tensor.stride_col,
                    batch_nth,
                    tensor.line_size,
                    tensor.packing,
                    tensor.config,
                );
                tensor.buffer.view(layout)
            }
            BatchedMatrix::Viewed(view) => {
                let shape = view.shape();
                view.view(SliceIndex::new(batch_nth, shape))
            }
        }
    }
}

#[derive(CubeLaunch, CubeType)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<Lhs: Numeric, Rhs: Numeric, Acc: Numeric> {
    /// The lhs tensor.
    pub lhs: BatchedMatrix<Lhs>,
    /// The rhs tensor.
    pub rhs: BatchedMatrix<Rhs>,
    /// The tensor for loading the accumulator, if present
    pub acc: CubeOption<BatchedMatrix<Acc>>,
}

pub type TensorOutput<EO> = View<Line<EO>, Coords3d, ReadWrite>;

impl<Lhs: Numeric, Rhs: Numeric, Acc: Numeric> ConcreteInputsFactory
    for TensorInputs<Lhs, Rhs, Acc>
{
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        _selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        config: impl BatchConfig,
    ) -> Self::RuntimeArg<'a, R> {
        let config = config.global_config();
        let view = |handle: &'a MatmulInputHandleRef<'a, R>, ident, line_size| match handle {
            MatmulInputHandleRef::Normal(handle) => {
                let layout = BatchedGlobalLayoutLaunch::from_handle(
                    client,
                    handle,
                    problem,
                    line_size,
                    config.global_memory_config(ident).into(),
                );
                ViewArg::new::<BatchedGlobalLayout>(handle.as_array_arg(line_size), layout)
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => {
                let (data_layout, scales_layout) = BatchedGlobalLayoutLaunch::from_quantized_handle(
                    client,
                    data,
                    scale,
                    shape,
                    problem,
                    **scheme,
                    line_size,
                    config.global_memory_config(ident).into(),
                );
                let data_view =
                    ViewArg::new::<BatchedGlobalLayout>(data.as_array_arg(line_size), data_layout);
                let scales_view =
                    ViewArg::new::<BatchedGlobalScaleLayout>(scale.as_array_arg(1), scales_layout);
                ViewArg::new_quantized(data_view, scales_view, **scheme)
            }
        };

        TensorInputsLaunch::new(
            BatchedMatrixArgs::Viewed(view(lhs, MatmulIdent::Lhs, line_sizes.lhs)),
            BatchedMatrixArgs::Viewed(view(rhs, MatmulIdent::Rhs, line_sizes.rhs)),
            CubeOptionArgs::None,
        )
    }
}

impl<EG: Numeric> ConcreteOutputFactory for View<Line<EG>, Coords3d, ReadWrite> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        out: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        config: impl BatchConfig,
    ) -> Self::RuntimeArg<'a, R> {
        let config = config.global_config();
        let layout = BatchedGlobalLayoutLaunch::from_handle(
            client,
            out,
            problem,
            line_sizes.out,
            config.global_memory_config(MatmulIdent::Out).into(),
        );
        ViewArg::new::<BatchedGlobalLayout>(out.as_array_arg(line_sizes.out), layout)
    }
}

#[cube]
impl MatmulArgs for TensorArgs {
    type Output<EO: Numeric> = TensorOutput<EO>;
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorInputs<Lhs, Rhs, EO>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = (
        BatchedMatrix<Lhs>,
        BatchedMatrix<Rhs>,
        CubeOption<BatchedMatrix<EO>>,
        View<Line<EO>, Coords3d, ReadWrite>,
    );

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric, G: GlobalConfig>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        #[comptime] _config: G,
    ) -> Self::State<Lhs, Rhs, EO> {
        (
            input.lhs.cloned(),
            input.rhs.cloned(),
            match &input.acc {
                CubeOption::Some(val) => CubeOption::new_Some(val.cloned()),
                CubeOption::None => CubeOption::new_None(),
            },
            *output,
        )
    }

    fn get_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> BatchedMatrix<Lhs> {
        state.0.cloned()
    }

    fn get_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> BatchedMatrix<Rhs> {
        state.1.cloned()
    }

    fn get_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<BatchedMatrix<EO>> {
        match &state.2 {
            CubeOption::Some(val) => CubeOption::new_Some(val.cloned()),
            CubeOption::None => CubeOption::new_None(),
        }
    }

    fn get_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        state.3
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensor maps.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorMapArgs;

#[derive(CubeLaunch, CubeType)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorMapInputs<Lhs: Numeric, Rhs: Numeric, EO: Numeric> {
    /// The lhs tensor.
    pub lhs: View<Line<Lhs>, Coords3d>,
    /// The rhs tensor.
    pub rhs: View<Line<Rhs>, Coords3d>,
    /// The accumulator
    pub acc: CubeOption<View<Line<EO>, Coords3d>>,
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory
    for TensorMapInputs<Lhs, Rhs, EO>
{
    fn create<'a, R: Runtime>(
        _client: &ComputeClient<R::Server>,
        lhs_handle: &'a MatmulInputHandleRef<'a, R>,
        rhs_handle: &'a MatmulInputHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        _config: impl BatchConfig,
    ) -> Self::RuntimeArg<'a, R> {
        let lhs = lhs_handle.data();
        let rhs = rhs_handle.data();

        let tiling_scheme = selection.tiling_scheme;
        let stage_m = tiling_scheme.elements_in_stage_m();
        let stage_n = tiling_scheme.elements_in_stage_n();
        let stage_k = tiling_scheme.elements_in_stage_k();
        let stage_size_lhs = match problem.lhs_layout {
            components::MatrixLayout::RowMajor => {
                vec![1, stage_m, tiling_scheme.elements_in_tile_k()]
            }
            components::MatrixLayout::ColMajor => {
                vec![1, stage_k, tiling_scheme.elements_in_tile_m()]
            }
        };
        let stage_size_rhs = match problem.rhs_layout {
            components::MatrixLayout::RowMajor => {
                vec![1, stage_k, tiling_scheme.elements_in_tile_n()]
            }
            components::MatrixLayout::ColMajor => {
                vec![1, stage_n, tiling_scheme.elements_in_tile_k()]
            }
        };

        let lhs_elem_size = size_of::<Lhs>();
        let rhs_elem_size = size_of::<Rhs>();

        let lhs_rank = lhs.shape.len();
        let mut lhs_shape = vec![
            problem.lhs_batches[0],
            lhs.shape[lhs_rank - 2],
            lhs.shape[lhs_rank - 1],
        ];
        let mut lhs_strides = if lhs_rank > 2 {
            lhs.strides[lhs_rank - 3..].to_vec()
        } else {
            vec![1, lhs.strides[lhs_rank - 2], lhs.strides[lhs_rank - 1]]
        };

        let rhs_rank = rhs.shape.len();
        let mut rhs_shape = vec![
            problem.rhs_batches[0],
            rhs.shape[rhs_rank - 2],
            rhs.shape[rhs_rank - 1],
        ];
        let mut rhs_strides = if rhs_rank > 2 {
            rhs.strides[rhs_rank - 3..].to_vec()
        } else {
            vec![1, rhs.strides[rhs_rank - 2], rhs.strides[rhs_rank - 1]]
        };

        let mut lhs_transposed = false;
        let mut rhs_transposed = false;

        // TMA assumes the last stride is contiguous and won't even take it, so we need to map it
        // with transposed shape and stride. Tensor metadata still has the normal layout.
        if matches!(problem.lhs_layout, components::MatrixLayout::ColMajor) {
            lhs_shape.swap(lhs_rank - 1, lhs_rank - 2);
            lhs_strides.swap(lhs_rank - 1, lhs_rank - 2);
            lhs_transposed = true;
        }
        if matches!(problem.rhs_layout, components::MatrixLayout::ColMajor) {
            rhs_shape.swap(rhs_rank - 1, rhs_rank - 2);
            rhs_strides.swap(rhs_rank - 1, rhs_rank - 2);
            rhs_transposed = true;
        }

        fn prefetch(bytes: usize) -> TensorMapPrefetch {
            match bytes {
                ..64 => TensorMapPrefetch::None,
                64..128 => TensorMapPrefetch::B64,
                128..256 => TensorMapPrefetch::B128,
                256.. => TensorMapPrefetch::B256,
            }
        }

        let prefetch_lhs = prefetch(stage_size_lhs[2] as usize * lhs_elem_size);
        let prefetch_rhs = prefetch(stage_size_rhs[2] as usize * rhs_elem_size);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if TypeId::of::<Lhs>() == TypeId::of::<f32>() {
            tf32::as_type_native_unchecked()
        } else {
            Lhs::as_type_native_unchecked()
        };
        let rhs_elem = if TypeId::of::<Rhs>() == TypeId::of::<f32>() {
            tf32::as_type_native_unchecked()
        } else {
            Rhs::as_type_native_unchecked()
        };

        let meta_lhs = TensorMapMeta {
            format: TensorMapFormat::Tiled {
                tile_size: stage_size_lhs,
            },
            rank: 3,
            shape: lhs_shape.clone(),
            strides: lhs_strides,
            elem_stride: vec![1, 1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            prefetch: prefetch_lhs,
            oob_fill: OobFill::Zero,
            storage_ty: lhs_elem,
        };

        let meta_rhs = TensorMapMeta {
            format: TensorMapFormat::Tiled {
                tile_size: stage_size_rhs,
            },
            rank: 3,
            shape: rhs_shape.clone(),
            strides: rhs_strides,
            elem_stride: vec![1, 1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            prefetch: prefetch_rhs,
            oob_fill: OobFill::Zero,
            storage_ty: rhs_elem,
        };

        let lhs = TensorMapArg {
            tensor: lhs.as_tensor_arg(line_sizes.lhs),
            metadata: meta_lhs,
        };
        let rhs = TensorMapArg {
            tensor: rhs.as_tensor_arg(line_sizes.rhs),
            metadata: meta_rhs,
        };

        let view = |buffer, shape: &[usize], transposed| {
            let batches = ScalarArg::new(shape[0] as u32);
            let (rows, cols) = match transposed {
                true => (
                    ScalarArg::new(shape[2] as u32),
                    ScalarArg::new(shape[1] as u32),
                ),
                false => (
                    ScalarArg::new(shape[1] as u32),
                    ScalarArg::new(shape[2] as u32),
                ),
            };
            let shape = (batches, rows, cols);
            let layout = SimpleTmaGlobalLayoutLaunch::new(transposed, shape);
            ViewArg::new_tensor_map::<SimpleTmaGlobalLayout>(buffer, layout)
        };

        TensorMapInputsLaunch::new(
            view(lhs, &lhs_shape, lhs_transposed),
            view(rhs, &rhs_shape, rhs_transposed),
            CubeOptionArgs::None,
        )
    }
}

#[cube]
impl MatmulArgs for TensorMapArgs {
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorMapInputs<Lhs, Rhs, EO>;
    type Output<EO: Numeric> = TensorOutput<EO>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = (
        BatchedMatrix<Lhs>,
        BatchedMatrix<Rhs>,
        CubeOption<BatchedMatrix<EO>>,
        View<Line<EO>, Coords3d, ReadWrite>,
    );

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric, G: GlobalConfig>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        #[comptime] _config: G,
    ) -> Self::State<Lhs, Rhs, EO> {
        (
            BatchedMatrix::new_Viewed(input.lhs),
            BatchedMatrix::new_Viewed(input.rhs),
            match &input.acc {
                CubeOption::Some(val) => CubeOption::new_Some(BatchedMatrix::new_Viewed(*val)),
                CubeOption::None => CubeOption::new_None(),
            },
            *output,
        )
    }

    fn get_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> BatchedMatrix<Lhs> {
        state.0.cloned()
    }

    fn get_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> BatchedMatrix<Rhs> {
        state.1.cloned()
    }

    fn get_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<BatchedMatrix<EO>> {
        match &state.2 {
            CubeOption::Some(val) => CubeOption::new_Some(val.cloned()),
            CubeOption::None => CubeOption::new_None(),
        }
    }

    fn get_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        state.3
    }
}
