use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::{
    CubeOptionArgs, FastDivmod, FastDivmodArgs,
    tensor::{
        launch::ViewArg,
        layout::{
            VirtualLayoutLaunch,
            chain::{Chain, ChainLaunch},
        },
    },
};

use crate::{
    components::{
        ConvGemmConfig, ConvolutionProblem,
        global::{
            layout::{
                BiasLayout, BiasLayoutLaunch, Im2colLayout, Im2colLayoutLaunch, NhwcLayout,
                NhwcLayoutLaunch, OutLayout, OutLayoutLaunch, WeightLayout, WeightLayoutLaunch,
            },
            read::layout::{
                TmaDummyLayout, TmaDummyLayoutLaunch, TmaWeightLayout, TmaWeightLayoutLaunch,
            },
        },
    },
    kernels::layered::algorithm::simple_tma::{calculate_lower_corner, calculate_upper_corner},
};
use cubecl_matmul::{
    MatmulInputHandleRef,
    components::{
        MatmulElems, MatmulLineSizes, MatmulSelection,
        global::{
            GlobalConfig,
            args::{
                TensorInputs, TensorInputsLaunch, TensorMapInputs, TensorMapInputsLaunch,
                TensorOutput, TensorOutputLaunch,
            },
            memory::{NoopLayout, NoopLayoutLaunch},
        },
        stage::StageConfig as _,
    },
};

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub shape_m: u32,
    pub shape_n: u32,
    pub shape_k: u32,
    pub channels: u32,
    pub padded_channels: FastDivmod,
    pub shape_out: Sequence<FastDivmod>,
}

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a TensorHandleRef<'a, R>>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R>;
}

/// Create the output runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R>;
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory for TensorInputs<Lhs, Rhs, EO> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a TensorHandleRef<'a, R>>,
        _selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R> {
        type LhsLayout = Chain<NhwcLayout, Im2colLayout>;
        type RhsLayout = Chain<NhwcLayout, WeightLayout>;

        let load_width = client.properties().hardware.load_width;
        let channel_align = load_width as usize / dtypes.lhs_global.size_bits();

        let layout_nhwc = |handle, line_size, check_spatial| {
            NhwcLayoutLaunch::from_handle(
                handle,
                line_size as u32,
                check_spatial,
                !problem.channels.is_multiple_of(channel_align),
            )
        };
        let layout_lhs = Im2colLayoutLaunch::from_args(
            client,
            problem,
            config.convolution_params(),
            config.lhs_global_memory_config(),
            dtypes,
        );
        let layout_rhs = WeightLayoutLaunch::from_args(
            client,
            problem,
            config.convolution_params(),
            config.rhs_global_memory_config(),
            dtypes,
        );
        let layout_bias =
            BiasLayoutLaunch::new(ScalarArg::new(problem.n as u32), line_sizes.out as u32);

        let layout_lhs = {
            let global = layout_nhwc(lhs.data(), line_sizes.lhs, config.check_spatial_bounds());
            ChainLaunch::new(global, layout_lhs)
        };
        let layout_rhs = {
            let global = layout_nhwc(rhs.data(), line_sizes.rhs, false);
            ChainLaunch::new(global, layout_rhs)
        };

        TensorInputsLaunch::new(
            ViewArg::new::<LhsLayout>(lhs.data().as_array_arg(line_sizes.lhs), layout_lhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            ViewArg::new::<RhsLayout>(rhs.data().as_array_arg(line_sizes.rhs), layout_rhs),
            VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()),
            bias.map(|bias| {
                ViewArg::new::<BiasLayout>(bias.as_array_arg(line_sizes.out), layout_bias)
            })
            .into(),
            bias.map(|_| VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new()))
                .into(),
        )
    }
}

impl<EG: Numeric> ConcreteOutputFactory for TensorOutput<EG> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        out: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        _dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R> {
        type Layout = Chain<NhwcLayout, OutLayout>;

        let global = NhwcLayoutLaunch::from_handle(out, line_sizes.out as u32, false, false);
        let layout = OutLayoutLaunch::from_args(client, problem, config.out_global_memory_config());
        let layout = ChainLaunch::new(global, layout);
        let view = ViewArg::new::<Layout>(out.as_array_arg(line_sizes.out), layout);
        let batch = VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new());
        TensorOutputLaunch::new(view, batch)
    }
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory
    for TensorMapInputs<Lhs, Rhs, EO>
{
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a TensorHandleRef<'a, R>>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg<'a, R> {
        let tiling_scheme = selection.tiling_scheme;
        let stage_m = tiling_scheme.elements_per_stage_along_m();
        let stage_n = tiling_scheme.elements_per_stage_along_n();
        let tile_size_k = tiling_scheme.tile_size.k;

        let mut stage_size_rhs = vec![1; problem.dimensionality.num_dims() as usize];
        stage_size_rhs.insert(0, stage_n);
        stage_size_rhs.push(tile_size_k);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if *dtypes.lhs_stage == f32::as_type_native_unchecked() {
            tf32::as_type_native_unchecked()
        } else {
            *dtypes.lhs_stage
        };

        let mut elem_stride = vec![1; 2 + problem.stride.len()];

        for (i, stride) in problem.stride.iter().enumerate() {
            elem_stride[i + 1] = *stride as usize;
        }

        let lhs = TensorMapArg::new(
            TensorMapFormat::Im2col {
                pixel_box_lower_corner: calculate_lower_corner(&problem.padding),
                pixel_box_upper_corner: calculate_upper_corner(
                    &problem.padding,
                    &problem.kernel_size,
                    &problem.dilation,
                ),
                channels_per_pixel: tile_size_k,
                pixels_per_column: stage_m,
            },
            lhs.data().as_tensor_arg(line_sizes.lhs),
            lhs_elem,
        )
        .with_elem_stride(elem_stride);

        let rhs = TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: stage_size_rhs,
            },
            rhs.data().as_tensor_arg(1),
            *dtypes.rhs_global,
        );

        let padded_channels = (problem.channels as u32)
            .next_multiple_of(config.matmul_config().stage_config().elements_in_tile_k());

        // Dummy layout since we don't support im2col loading rn
        let lhs_layout = TmaDummyLayoutLaunch::new();
        let rhs_layout = TmaWeightLayoutLaunch::new(
            FastDivmodArgs::new(client, padded_channels),
            problem.kernel_size.clone(),
        );

        let bias = bias.map(|bias| {
            let layout =
                BiasLayoutLaunch::new(ScalarArg::new(problem.n as u32), line_sizes.out as u32);
            ViewArg::new::<BiasLayout>(bias.as_array_arg(line_sizes.out), layout)
        });

        TensorMapInputsLaunch::new(
            ViewArg::new_tensor_map::<TmaDummyLayout>(lhs, lhs_layout),
            ViewArg::new_tensor_map::<TmaWeightLayout>(rhs, rhs_layout),
            bias.into(),
            CubeOptionArgs::Some(VirtualLayoutLaunch::new::<NoopLayout>(
                NoopLayoutLaunch::new(),
            )),
        )
    }
}
