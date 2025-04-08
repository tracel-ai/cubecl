use std::marker::PhantomData;

use cubecl_core::{CubeCount, CubeDim, Runtime, prelude::TensorHandleRef};

use crate::{
    convolution::{
        base::{ConvolutionConfigFactory, ConvolutionProblem},
        homogeneous::simple_tma::SimpleTmaConvolutionFamily,
    },
    matmul::components::{
        InvalidConfigError, MatmulSelection, global::args::TensorMapArgs, stage,
        tile::TileMatmulFamily,
    },
};

use super::Algorithm;

pub const TMA_STRIDE_ALIGN: usize = 16;

/// Cmma convolution
pub struct SimpleTmaConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for SimpleTmaConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = stage::multi_buffer::MultiBufferMatmulFamily<Self::TileMatmul>;
    type GlobalConvolution = SimpleTmaConvolutionFamily<Self::StageMatmul>;

    type Args = TensorMapArgs;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim {
        CubeDim::new(selection.plane_dim, selection.tile_count.m, 1)
    }

    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn make_config(
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        check_problem_tma(problem)?;

        let config = Self::GlobalConvolution::make_config(input, problem, cube_dim, cube_count);
        Self::GlobalConvolution::check_config(&config)?;
        Ok(config)
    }

    fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>) -> bool {
        let stride_align = TMA_STRIDE_ALIGN / handle.elem_size;

        let mut strides = handle.strides.to_vec();
        strides.sort();

        // Permuted strides
        if handle.strides != strides {
            return false;
        }

        let aligned = handle.strides[..3]
            .iter()
            .all(|stride| stride % stride_align == 0);

        // channels doesn't need to be contiguous with the rest of the tensor
        strides[2] * handle.shape[2] == strides[1]
            && strides[1] * handle.shape[1] == handle.strides[0]
            && aligned
    }
}

fn check_problem_tma(problem: &ConvolutionProblem) -> Result<(), InvalidConfigError> {
    fn check_range(
        value: isize,
        name: &str,
        min: isize,
        max: isize,
    ) -> Result<(), InvalidConfigError> {
        if value < min || value > max {
            Err(Box::new(format!(
                "value {name} outside of valid range ({min}, {max})"
            )))
        } else {
            Ok(())
        }
    }

    let corner = calculate_upper_corner(problem.padding, problem.kernel_size, problem.dilation);
    check_range(corner[0] as isize, "corner_h", -128, 127)?;
    check_range(corner[1] as isize, "corner_w", -128, 127)?;

    let offset_h = (problem.kernel_size.0 - 1) * problem.dilation.0;
    let offset_w = (problem.kernel_size.1 - 1) * problem.dilation.1;
    check_range(offset_h as isize, "kernel size h", 0, 255)?;
    check_range(offset_w as isize, "kernel size w", 0, 255)?;

    check_range(problem.stride.0 as isize, "stride_h", 1, 8)?;
    check_range(problem.stride.1 as isize, "stride_w", 1, 8)?;

    Ok(())
}

pub fn calculate_upper_corner(
    padding: (i32, i32),
    kernel_size: (u32, u32),
    dilation: (u32, u32),
) -> Vec<i32> {
    let corner_h = padding.0 - (kernel_size.0 - 1) as i32 * dilation.0 as i32;
    let corner_w = padding.1 - (kernel_size.1 - 1) as i32 * dilation.1 as i32;

    vec![corner_h, corner_w]
}
