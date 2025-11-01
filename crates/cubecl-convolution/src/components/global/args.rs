use std::any::TypeId;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{
    CubeOptionArgs, FastDivmodArgs,
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
        MatmulIdent, MatmulLineSizes, MatmulSelection,
        global::{
            args::{
                TensorInputs, TensorInputsLaunch, TensorMapInputs, TensorMapInputsLaunch,
                TensorOutput, TensorOutputLaunch,
            },
            memory::{NoopLayout, NoopLayoutLaunch},
        },
    },
};

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a TensorHandleRef<'a, R>>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
    ) -> Self::RuntimeArg<'a, R>;
}

/// Create the output runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        out: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
    ) -> Self::RuntimeArg<'a, R>;
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric> ConcreteInputsFactory for TensorInputs<Lhs, Rhs, EO> {
    fn create<'a, R: Runtime>(
        client: &ComputeClient<R::Server>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a TensorHandleRef<'a, R>>,
        _selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
    ) -> Self::RuntimeArg<'a, R> {
        type LhsLayout = Chain<NhwcLayout, Im2colLayout>;
        type RhsLayout = Chain<NhwcLayout, WeightLayout>;

        let layout_nhwc = |handle, line_size, check| {
            NhwcLayoutLaunch::from_handle(handle, line_size as u32, check)
        };
        let layout_lhs = Im2colLayoutLaunch::from_args(
            client,
            problem,
            config.convolution_params(),
            config.global_memory_config(MatmulIdent::Lhs),
        );
        let layout_rhs = WeightLayoutLaunch::from_args(
            client,
            problem,
            config.convolution_params(),
            config.global_memory_config(MatmulIdent::Rhs),
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
        client: &ComputeClient<R::Server>,
        out: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
    ) -> Self::RuntimeArg<'a, R> {
        type Layout = Chain<NhwcLayout, OutLayout>;

        let global = NhwcLayoutLaunch::from_handle(out, line_sizes.out as u32, false);
        let layout = OutLayoutLaunch::from_args(
            client,
            problem,
            config.global_memory_config(MatmulIdent::Out),
        );
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
        client: &ComputeClient<R::Server>,
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        bias: Option<&'a TensorHandleRef<'a, R>>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        config: impl ConvGemmConfig,
    ) -> Self::RuntimeArg<'a, R> {
        let tiling_scheme = selection.tiling_scheme;
        let stage_m = tiling_scheme.elements_in_stage_m();
        let stage_n = tiling_scheme.elements_in_stage_n();
        let tile_size_k = tiling_scheme.elements_in_tile_k();
        let stage_size_rhs = vec![stage_n, 1, tile_size_k];

        let lhs_elem_size = size_of::<Lhs>();
        let rhs_elem_size = size_of::<Rhs>();

        fn prefetch(bytes: usize) -> TensorMapPrefetch {
            match bytes {
                ..64 => TensorMapPrefetch::None,
                64..128 => TensorMapPrefetch::B64,
                128..256 => TensorMapPrefetch::B128,
                256.. => TensorMapPrefetch::B256,
            }
        }

        let prefetch_lhs = prefetch(tile_size_k as usize * lhs_elem_size);
        let prefetch_rhs = prefetch(stage_size_rhs[2] as usize * rhs_elem_size);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if TypeId::of::<Lhs>() == TypeId::of::<f32>() {
            tf32::as_type_native_unchecked()
        } else {
            Lhs::as_type_native_unchecked()
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
        .with_elem_stride(elem_stride)
        .with_prefetch(prefetch_lhs);

        let rhs = TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: stage_size_rhs,
            },
            rhs.data().as_tensor_arg(1),
            Rhs::as_type_native_unchecked(),
        )
        .with_prefetch(prefetch_rhs);

        let padded_channels =
            (problem.channels as u32).next_multiple_of(config.tiling_scheme().elements_in_tile_k());

        // Dummy layout since we don't support im2col loading rn
        let lhs_layout = TmaDummyLayoutLaunch::new();
        let rhs_layout = TmaWeightLayoutLaunch::new(FastDivmodArgs::new(client, padded_channels));

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
