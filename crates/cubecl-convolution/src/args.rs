use std::any::TypeId;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::algorithm::simple_tma::{calculate_lower_corner, calculate_upper_corner};
use cubecl_matmul::components::{
    MatmulLineSizes, MatmulSelection,
    global::args::{TensorInputs, TensorInputsLaunch, TensorMapInputs, TensorMapInputsLaunch},
};

use super::base::ConvolutionProblem;

pub trait ConvInputsLaunch: LaunchArg {
    fn create<'a, R: Runtime>(
        lhs: &'a TensorHandleRef<'a, R>,
        rhs: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R>;
}

impl<EI: Numeric> ConvInputsLaunch for TensorInputs<EI> {
    fn create<'a, R: Runtime>(
        lhs: &'a TensorHandleRef<'a, R>,
        rhs: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        _problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        TensorInputsLaunch::new(
            lhs.as_tensor_arg(line_sizes.lhs),
            None.into(),
            rhs.as_tensor_arg(line_sizes.rhs),
            None.into(),
        )
    }
}

impl<EI: Numeric> ConvInputsLaunch for TensorMapInputs<EI> {
    fn create<'a, R: Runtime>(
        lhs: &'a TensorHandleRef<'a, R>,
        rhs: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        let tiling_scheme = selection.tiling_scheme;
        let stage_m = tiling_scheme.elements_in_stage_m();
        let stage_n = tiling_scheme.elements_in_stage_n();
        let tile_size_k = tiling_scheme.elements_in_tile_k();
        let stage_size_rhs = vec![stage_n, 1, tile_size_k];

        let elem_size = size_of::<EI>();

        fn prefetch(bytes: usize) -> TensorMapPrefetch {
            match bytes {
                ..64 => TensorMapPrefetch::None,
                64..128 => TensorMapPrefetch::B64,
                128..256 => TensorMapPrefetch::B128,
                256.. => TensorMapPrefetch::B256,
            }
        }

        let prefetch_lhs = prefetch(tile_size_k as usize * elem_size);
        let prefetch_rhs = prefetch(stage_size_rhs[2] as usize * elem_size);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let elem = if TypeId::of::<EI>() == TypeId::of::<f32>() {
            tf32::as_elem_native_unchecked()
        } else {
            EI::as_elem_native_unchecked()
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
            lhs.as_tensor_arg(line_sizes.lhs),
            elem,
        )
        .with_elem_stride(elem_stride)
        .with_prefetch(prefetch_lhs);

        let rhs = TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: stage_size_rhs,
            },
            rhs.as_tensor_arg(1),
            EI::as_elem_native_unchecked(),
        )
        .with_prefetch(prefetch_rhs);

        TensorMapInputsLaunch::new(lhs, rhs)
    }
}
