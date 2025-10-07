use std::any::TypeId;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, server::TensorMapMeta, unexpanded};
use cubecl_std::{
    CubeOption, CubeOptionArgs, CubeOptionExpand,
    tensor::{
        View,
        layout::Coords3d,
        r#virtual::{VirtualTensorOperations, VirtualTensorOperationsExpand},
    },
};

use crate::{
    MatmulInputHandleRef,
    components::{self, MatmulLineSizes, MatmulProblem, MatmulSelection},
};

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R>;
}

/// Create the output runtime argument for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        out: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
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
    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<Lhs, Rhs, EO>;

    fn view_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Lhs>, Coords3d> {
        unexpanded!()
    }
    fn view_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Line<Rhs>, Coords3d> {
        unexpanded!()
    }
    fn view_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<View<Line<EO>, Coords3d>> {
        unexpanded!()
    }
    fn view_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
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
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<Lhs: Numeric, Rhs: Numeric, Acc: Numeric> {
    /// The lhs tensor.
    pub lhs: Tensor<Line<Lhs>>,
    pub lhs_scale: CubeOption<Tensor<f32>>,
    /// The rhs tensor.
    pub rhs: Tensor<Line<Rhs>>,
    pub rhs_scale: CubeOption<Tensor<f32>>,
    /// The tensor for loading the accumulator, if present
    pub acc: CubeOption<Tensor<Line<Acc>>>,
}

impl<Lhs: Numeric, Rhs: Numeric, Acc: Numeric> ConcreteInputsFactory
    for TensorInputs<Lhs, Rhs, Acc>
{
    fn create<'a, R: Runtime>(
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        _selection: &MatmulSelection,
        _problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        TensorInputsLaunch::new(
            lhs.data().as_tensor_arg(line_sizes.lhs),
            lhs.scale().map(|it| it.as_tensor_arg(1)).into(),
            rhs.data().as_tensor_arg(line_sizes.rhs),
            rhs.scale().map(|it| it.as_tensor_arg(1)).into(),
            CubeOptionArgs::None,
        )
    }
}

impl<EG: Numeric> ConcreteOutputFactory for Tensor<Line<EG>> {
    fn create<'a, R: Runtime>(
        out: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        _problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        out.as_tensor_arg(line_sizes.out)
    }
}

#[cube]
impl MatmulArgs for TensorArgs {
    type Output<EO: Numeric> = Tensor<Line<EO>>;
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorInputs<Lhs, Rhs, EO>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = (
        *const Tensor<Line<Lhs>>,
        *const Tensor<Line<Rhs>>,
        CubeOption<*const Tensor<Line<EO>>>,
        *mut Tensor<Line<EO>>,
        CubeOption<f32>,
        CubeOption<f32>,
    );

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<Lhs, Rhs, EO> {
        let lhs_scale = match &input.lhs_scale {
            CubeOption::Some(scale) => CubeOption::new_Some(scale[0]),
            CubeOption::None => CubeOption::new_None(),
        };
        let rhs_scale = match &input.rhs_scale {
            CubeOption::Some(scale) => CubeOption::new_Some(scale[0]),
            CubeOption::None => CubeOption::new_None(),
        };
        let acc = match &input.acc {
            CubeOption::None => CubeOption::new_None(),
            CubeOption::Some(acc) => {
                let ptr: *const Tensor<Line<EO>> = acc;
                CubeOption::new_Some(ptr)
            }
        };
        (&input.lhs, &input.rhs, acc, output, lhs_scale, rhs_scale)
    }

    fn has_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<()> {
        match state.2 {
            CubeOption::None => CubeOption::new_None(),
            CubeOption::Some(_) => CubeOption::new_Some(()),
        }
    }

    fn read_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<Lhs> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<Rhs> {
        unsafe { (*state.1)[coordinate] }
    }

    fn read_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<EO> {
        unsafe { (*state.2.unwrap())[coordinate] }
    }

    fn read_window_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Lhs>> {
        unsafe { (*state.0).slice(start, end) }
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Rhs>> {
        unsafe { (*state.1).slice(start, end) }
    }

    fn read_window_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EO>> {
        unsafe { (*state.2.unwrap()).slice(start, end) }
    }

    fn as_tensor_map_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<TensorMap<Lhs>> {
        CubeOption::new_None()
    }

    fn as_tensor_map_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<TensorMap<Rhs>> {
        CubeOption::new_None()
    }

    fn as_tensor_map_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<TensorMap<EO>> {
        CubeOption::new_None()
    }

    fn shape_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.2.unwrap()).shape(dim) }
    }

    fn shape_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.3).shape(dim) }
    }

    fn stride_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.2.unwrap()).stride(dim) }
    }

    fn stride_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.3).stride(dim) }
    }

    fn write_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
        value: Line<EO>,
    ) {
        unsafe { (*state.3)[coordinate] = value }
    }

    fn rank_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2.unwrap()).rank() }
    }

    fn rank_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.3).rank() }
    }

    fn len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.0).len() }
    }

    fn len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.1).len() }
    }

    fn len_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2.unwrap()).len() }
    }

    fn len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.3).len() }
    }

    fn buffer_len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.0).buffer_len() }
    }

    fn buffer_len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.1).buffer_len() }
    }

    fn buffer_len_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.2.unwrap()).buffer_len() }
    }

    fn buffer_len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.3).buffer_len() }
    }

    fn line_size_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.0).line_size() }
    }
    fn line_size_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.1).line_size() }
    }

    #[allow(unused_variables)]
    fn line_size_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            match state.2 {
                CubeOptionExpand::None => 1,
                CubeOptionExpand::Some(t) => t.__expand_line_size_method(scope),
            }
        })
    }

    fn line_size_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.3).line_size() }
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
    pub lhs: TensorMap<Lhs>,
    /// The rhs tensor.
    pub rhs: TensorMap<Rhs>,
    /// The accumulator
    pub acc: CubeOption<Tensor<Line<EO>>>,
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory
    for TensorMapInputs<Lhs, Rhs, EO>
{
    fn create<'a, R: Runtime>(
        lhs_handle: &'a MatmulInputHandleRef<'a, R>,
        rhs_handle: &'a MatmulInputHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
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

        // TMA assumes the last stride is contiguous and won't even take it, so we need to map it
        // with transposed shape and stride. Tensor metadata still has the normal layout.
        if matches!(problem.lhs_layout, components::MatrixLayout::ColMajor) {
            lhs_shape.swap(lhs_rank - 1, lhs_rank - 2);
            lhs_strides.swap(lhs_rank - 1, lhs_rank - 2);
        }
        if matches!(problem.rhs_layout, components::MatrixLayout::ColMajor) {
            rhs_shape.swap(rhs_rank - 1, rhs_rank - 2);
            rhs_strides.swap(rhs_rank - 1, rhs_rank - 2);
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
            shape: lhs_shape,
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
            shape: rhs_shape,
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

        TensorMapInputsLaunch::new(lhs, rhs, CubeOptionArgs::None)
    }
}

#[cube]
impl MatmulArgs for TensorMapArgs {
    type Input<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = TensorMapInputs<Lhs, Rhs, EO>;
    type Output<EO: Numeric> = Tensor<Line<EO>>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = (
        *const TensorMap<Lhs>,
        *const TensorMap<Rhs>,
        CubeOption<*const Tensor<Line<EO>>>,
        *mut Tensor<Line<EO>>,
    );

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<Lhs, Rhs, EO> {
        let acc = match &input.acc {
            CubeOption::None => CubeOption::new_None(),
            CubeOption::Some(acc) => {
                let ptr: *const Tensor<Line<EO>> = acc;
                CubeOption::new_Some(ptr)
            }
        };
        (&input.lhs, &input.rhs, acc, output)
    }

    fn has_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<()> {
        match state.2 {
            CubeOption::None => CubeOption::new_None(),
            CubeOption::Some(_) => CubeOption::new_Some(()),
        }
    }

    fn read_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _coordinate: u32,
    ) -> Line<Lhs> {
        unimplemented!("Can't directly read from TensorMap")
    }

    fn read_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _coordinate: u32,
    ) -> Line<Rhs> {
        unimplemented!("Can't directly read from TensorMap")
    }

    fn read_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<EO> {
        unsafe { (*state.2.unwrap())[coordinate] }
    }

    #[allow(unused)]
    fn read_window_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Lhs>> {
        unimplemented!("Can't directly read from TensorMap")
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    #[allow(unused)]
    fn read_window_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Rhs>> {
        unimplemented!("Can't directly read from TensorMap")
    }

    fn read_window_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EO>> {
        unsafe { (*state.2.unwrap()).slice(start, end) }
    }

    fn as_tensor_map_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<TensorMap<Lhs>> {
        CubeOption::new_Some(unsafe { *state.0 })
    }

    fn as_tensor_map_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<TensorMap<Rhs>> {
        CubeOption::new_Some(unsafe { *state.1 })
    }

    fn as_tensor_map_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> CubeOption<TensorMap<EO>> {
        CubeOption::new_None()
    }

    fn shape_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.2.unwrap()).shape(dim) }
    }

    fn shape_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.3 }.shape(dim)
    }

    fn stride_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.0 }.stride(dim)
    }

    fn stride_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.1 }.stride(dim)
    }

    fn stride_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.2.unwrap()).stride(dim) }
    }

    fn stride_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.3 }.stride(dim)
    }

    fn write_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
        value: Line<EO>,
    ) {
        unsafe { (*state.3)[coordinate] = value }
    }

    fn rank_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2.unwrap()).rank() }
    }

    fn rank_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.3).rank() }
    }

    fn len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.0).len() }
    }

    fn len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.1).len() }
    }

    fn len_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2.unwrap()).len() }
    }

    fn len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.3).len() }
    }

    fn buffer_len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.0).buffer_len() }
    }

    fn buffer_len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.1).buffer_len() }
    }

    fn buffer_len_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.2.unwrap()).buffer_len() }
    }

    fn buffer_len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.3).buffer_len() }
    }

    fn line_size_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        1
    }
    fn line_size_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        1
    }
    #[allow(unused_variables)]
    fn line_size_acc<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            match state.2 {
                CubeOptionExpand::None => 1,
                CubeOptionExpand::Some(t) => t.__expand_line_size_method(scope),
            }
        })
    }
    fn line_size_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.3).line_size() }
    }
}

mod __lhs {
    use super::*;

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeType
        for TensorLhs<Lhs, Rhs, EO, GA>
    {
        type ExpandType = TensorLhsExpand<Lhs, Rhs, EO, GA>;
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorLhsExpand<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> IntoMut
        for TensorLhsExpand<Lhs, Rhs, EO, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug
        for TensorLhsExpand<Lhs, Rhs, EO, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorLhs<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Copy for TensorLhs<Lhs, Rhs, EO, GA> {}
}

mod __rhs {
    use super::*;

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeType
        for TensorRhs<Lhs, Rhs, EO, GA>
    {
        type ExpandType = TensorRhsExpand<Lhs, Rhs, EO, GA>;
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorRhsExpand<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> IntoMut
        for TensorRhsExpand<Lhs, Rhs, EO, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug
        for TensorRhsExpand<Lhs, Rhs, EO, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorRhs<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Copy for TensorRhs<Lhs, Rhs, EO, GA> {}
}

mod __acc {
    use super::*;

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeType
        for TensorAcc<Lhs, Rhs, EO, GA>
    {
        type ExpandType = TensorAccExpand<Lhs, Rhs, EO, GA>;
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorAccExpand<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> IntoMut
        for TensorAccExpand<Lhs, Rhs, EO, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug
        for TensorAccExpand<Lhs, Rhs, EO, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorAcc<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Copy for TensorAcc<Lhs, Rhs, EO, GA> {}
}

mod __output {
    use super::*;

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeType
        for TensorOutput<Lhs, Rhs, EO, GA>
    {
        type ExpandType = TensorOutputExpand<Lhs, Rhs, EO, GA>;
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorOutput<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorOutputExpand<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> IntoMut
        for TensorOutputExpand<Lhs, Rhs, EO, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug
        for TensorOutputExpand<Lhs, Rhs, EO, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Copy
        for TensorOutput<Lhs, Rhs, EO, GA>
    {
    }
}
