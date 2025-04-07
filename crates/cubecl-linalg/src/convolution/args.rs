use std::any::TypeId;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, server::TensorMapMeta};

use crate::{
    convolution::algorithm::simple_tma::calculate_upper_corner,
    matmul::components::{
        MatmulSelection,
        global::args::{TensorInputs, TensorInputsLaunch, TensorMapInputs, TensorMapInputsLaunch},
    },
};

use super::base::ConvolutionProblem;

pub trait ConvInputsLaunch: LaunchArg {
    fn create<'a, R: Runtime>(
        lhs: &'a TensorHandleRef<'a, R>,
        rhs: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
    ) -> Self::RuntimeArg<'a, R>;
}

impl<EI: Numeric> ConvInputsLaunch for TensorInputs<EI> {
    fn create<'a, R: Runtime>(
        lhs: &'a TensorHandleRef<'a, R>,
        rhs: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        problem: &ConvolutionProblem,
    ) -> Self::RuntimeArg<'a, R> {
        TensorInputsLaunch::new(
            lhs.as_tensor_arg(problem.lhs_line_size),
            rhs.as_tensor_arg(problem.rhs_line_size),
        )
    }
}

impl<EI: Numeric> ConvInputsLaunch for TensorMapInputs<EI> {
    fn create<'a, R: Runtime>(
        lhs: &'a TensorHandleRef<'a, R>,
        rhs: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
    ) -> Self::RuntimeArg<'a, R> {
        let stage_m = selection.tile_count.m * selection.tile_shape.m;
        let stage_size_rhs = vec![1, selection.tile_shape.k, selection.tile_shape.n];

        let elem_size = size_of::<EI>();

        // Reset to 4D so we can pad the channels
        let rhs_shape = vec![
            problem.kernel_size.0 as usize * problem.kernel_size.1 as usize,
            problem.channels,
            problem.n,
        ];
        let rhs_strides = vec![
            rhs.strides[0] * problem.channels,
            rhs.strides[0],
            rhs.strides[1],
        ];

        fn prefetch(bytes: usize) -> TensorMapPrefetch {
            match bytes {
                ..64 => TensorMapPrefetch::None,
                64..128 => TensorMapPrefetch::B64,
                128..256 => TensorMapPrefetch::B128,
                256.. => TensorMapPrefetch::B256,
            }
        }

        let prefetch_lhs = prefetch(selection.tile_shape.k as usize * elem_size);
        let prefetch_rhs = prefetch(stage_size_rhs[2] as usize * elem_size);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let elem = if TypeId::of::<EI>() == TypeId::of::<f32>() {
            tf32::as_elem_native_unchecked()
        } else {
            EI::as_elem_native_unchecked()
        };

        let lhs = TensorMapArg::new(
            TensorMapFormat::Im2col {
                pixel_box_lower_corner: vec![-problem.padding.0, -problem.padding.1],
                pixel_box_upper_corner: calculate_upper_corner(
                    problem.padding,
                    problem.kernel_size,
                    problem.dilation,
                ),
                channels_per_pixel: selection.tile_shape.k,
                pixels_per_column: stage_m,
            },
            lhs.as_tensor_arg(problem.lhs_line_size),
            elem,
        )
        .with_elem_stride(vec![
            1,
            problem.stride.0 as usize,
            problem.stride.1 as usize,
            1,
        ])
        .with_prefetch(prefetch_lhs);

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
            elem,
        };

        let rhs = TensorMapArg {
            tensor: rhs.as_tensor_arg(problem.rhs_line_size),
            metadata: meta_rhs,
        };

        TensorMapInputsLaunch::new(lhs, rhs)
    }
}
