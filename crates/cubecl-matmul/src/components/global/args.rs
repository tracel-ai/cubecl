use std::any::TypeId;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, server::TensorMapMeta, unexpanded};
use cubecl_std::{
    CubeOption, CubeOptionArgs, CubeOptionExpand,
    tensor::{
        View,
        launch::ViewArg,
        layout::{Coords1d, Coords3d, VirtualLayout, VirtualLayoutLaunch},
    },
};

use crate::{
    MatmulInputHandleRef,
    components::{
        self, MatmulIdent, MatmulLineSizes, MatmulProblem, MatmulSelection,
        batch::BatchConfig,
        global::{
            GlobalConfig,
            memory::{
                BatchLayout, BatchLayoutLaunch, GlobalLayout, GlobalLayoutLaunch,
                GlobalScaleLayout, NoopLayout, NoopLayoutLaunch, SimpleTmaGlobalLayout,
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

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, Coords3d> {
        unexpanded!()
    }
    fn batch_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: u32,
    ) -> u32 {
        unexpanded!()
    }
    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, Coords3d> {
        unexpanded!()
    }
    fn batch_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: u32,
    ) -> u32 {
        unexpanded!()
    }
    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<View<Line<EO>, Coords3d>> {
        unexpanded!()
    }
    fn batch_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: u32,
    ) -> u32 {
        unexpanded!()
    }
    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        unexpanded!()
    }
    fn batch_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: u32,
    ) -> u32 {
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

#[derive(CubeLaunch, CubeType, Clone, Copy)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<Lhs: Numeric, Rhs: Numeric, Acc: Numeric> {
    /// The lhs tensor.
    lhs: View<Line<Lhs>, Coords3d>,
    lhs_batch: VirtualLayout<Coords1d, Coords1d>,
    /// The rhs tensor.
    rhs: View<Line<Rhs>, Coords3d>,
    rhs_batch: VirtualLayout<Coords1d, Coords1d>,
    /// The tensor for loading the accumulator, if present
    acc: CubeOption<View<Line<Acc>, Coords3d>>,
    acc_batch: CubeOption<VirtualLayout<Coords1d, Coords1d>>,
}

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
                let layout = GlobalLayoutLaunch::from_handle(
                    handle,
                    line_size,
                    config.global_memory_config(ident).into(),
                );
                ViewArg::new::<GlobalLayout>(handle.as_array_arg(line_size), layout)
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => {
                let (data_layout, scales_layout) = GlobalLayoutLaunch::from_quantized_handle(
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
                    ViewArg::new::<GlobalLayout>(data.as_array_arg(line_size), data_layout);
                let scales_view =
                    ViewArg::new::<GlobalScaleLayout>(scale.as_array_arg(1), scales_layout);
                ViewArg::new_quantized(data_view, scales_view, **scheme)
            }
        };
        let batch_layout = |handle: &'a MatmulInputHandleRef<'a, R>| match handle {
            MatmulInputHandleRef::Normal(handle) => {
                let layout = BatchLayoutLaunch::from_handle(client, handle, problem);
                VirtualLayoutLaunch::new::<BatchLayout>(layout)
            }
            MatmulInputHandleRef::Quantized { .. } => {
                VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new())
            }
        };

        TensorInputsLaunch::new(
            view(lhs, MatmulIdent::Lhs, line_sizes.lhs),
            batch_layout(lhs),
            view(rhs, MatmulIdent::Rhs, line_sizes.rhs),
            batch_layout(rhs),
            CubeOptionArgs::None,
            CubeOptionArgs::None,
        )
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct TensorOutput<EG: Numeric> {
    view: View<Line<EG>, Coords3d, ReadWrite>,
    batch: VirtualLayout<Coords1d, Coords1d>,
}

impl<EG: Numeric> ConcreteOutputFactory for TensorOutput<EG> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        out: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        config: impl BatchConfig,
    ) -> Self::RuntimeArg<'a, R> {
        let config = config.global_config();
        let layout = GlobalLayoutLaunch::from_handle(
            out,
            line_sizes.out,
            config.global_memory_config(MatmulIdent::Out).into(),
        );
        let batch = BatchLayoutLaunch::from_handle(client, out, problem);
        let view = ViewArg::new::<GlobalLayout>(out.as_array_arg(line_sizes.out), layout);
        TensorOutputLaunch::new(view, VirtualLayoutLaunch::new::<BatchLayout>(batch))
    }
}

#[cube]
impl MatmulArgs for TensorArgs {
    type Output<EO: Numeric> = TensorOutput<EO>;
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorInputs<Lhs, Rhs, EO>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> =
        (TensorInputs<Lhs, Rhs, EO>, TensorOutput<EO>);

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric, G: GlobalConfig>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        #[comptime] _config: G,
    ) -> Self::State<Lhs, Rhs, EO> {
        (*input, *output)
    }

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, Coords3d> {
        state.0.lhs
    }

    fn batch_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        state.0.lhs_batch.to_source_pos(batch)
    }

    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, Coords3d> {
        state.0.rhs
    }

    fn batch_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        state.0.rhs_batch.to_source_pos(batch)
    }

    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<View<Line<EO>, Coords3d>> {
        state.0.acc
    }

    fn batch_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        match state.0.acc_batch {
            CubeOption::Some(layout) => layout.to_source_pos(batch),
            CubeOption::None => batch,
        }
    }

    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        state.1.view
    }

    fn batch_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        state.1.batch.to_source_pos(batch)
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensor maps.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorMapArgs;

#[derive(CubeLaunch, CubeType, Clone, Copy)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorMapInputs<Lhs: Numeric, Rhs: Numeric, EO: Numeric> {
    /// The lhs tensor.
    pub lhs: View<Line<Lhs>, Coords3d>,
    /// The rhs tensor.
    pub rhs: View<Line<Rhs>, Coords3d>,
    /// The accumulator
    pub acc: CubeOption<View<Line<EO>, Coords3d>>,
    /// The accumulator batch layout
    pub acc_batch: CubeOption<VirtualLayout<Coords1d, Coords1d>>,
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
            CubeOptionArgs::None,
        )
    }
}

#[cube]
impl MatmulArgs for TensorMapArgs {
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorMapInputs<Lhs, Rhs, EO>;
    type Output<EO: Numeric> = TensorOutput<EO>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> =
        (TensorMapInputs<Lhs, Rhs, EO>, TensorOutput<EO>);

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric, G: GlobalConfig>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        #[comptime] _config: G,
    ) -> Self::State<Lhs, Rhs, EO> {
        (*input, *output)
    }

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, Coords3d> {
        state.0.lhs
    }

    fn batch_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        batch
    }

    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, Coords3d> {
        state.0.rhs
    }

    fn batch_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        batch
    }

    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<View<Line<EO>, Coords3d>> {
        state.0.acc
    }

    fn batch_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        match state.0.acc_batch {
            CubeOption::Some(layout) => layout.to_source_pos(batch),
            CubeOption::None => batch,
        }
    }

    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<EO>, Coords3d, ReadWrite> {
        state.1.view
    }

    fn batch_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: u32,
    ) -> u32 {
        state.1.batch.to_source_pos(batch)
    }
}
