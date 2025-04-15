use std::marker::PhantomData;

use cubecl_core::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
};

use crate::{
    convolution::{
        base::{ConvolutionConfigFactory, ConvolutionProblem},
        homogeneous::simple_tma::SimpleTmaConvolutionFamily,
    },
    matmul::components::{
        InputIdent, InvalidConfigError, MatmulSelection,
        global::args::TensorMapArgs,
        stage::{FullReaderFamily, plane_matmul::PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    tensor::{TensorHandle, into_contiguous_pitched},
};

use super::Algorithm;

pub const TMA_STRIDE_ALIGN: usize = 16;

/// Cmma convolution
pub struct SimpleTmaConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for SimpleTmaConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily>;
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

    fn check_availability<R: Runtime, MP: crate::matmul::components::MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::GlobalConvolution as ConvolutionConfigFactory>::Config,
    ) -> Result<(), crate::matmul::kernels::MatmulAvailabilityError> {
        <Self::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, MP>(
            client, config,
        )?;

        if !client
            .properties()
            .feature_enabled(cubecl_core::Feature::Tma(cubecl_core::TmaFeature::Base))
        {
            return Err(crate::matmul::kernels::MatmulAvailabilityError::TmaUnavailable);
        }

        Ok(())
    }

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: InputIdent,
    ) -> TensorHandle<R, E> {
        let mut handle = if has_valid_layout(handle, ident) {
            TensorHandle::from_ref(handle)
        } else {
            into_contiguous_pitched(client, handle)
        };
        match ident {
            InputIdent::Lhs => handle,
            InputIdent::Rhs => {
                handle.shape = vec![
                    handle.shape[0],
                    handle.shape[1] * handle.shape[2],
                    handle.shape[3],
                ];
                handle.strides = vec![handle.strides[0], handle.strides[2], handle.strides[3]];
                handle
            }
        }
    }
}

fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>, ident: InputIdent) -> bool {
    let stride_align = TMA_STRIDE_ALIGN / handle.elem_size;

    let aligned = handle.strides[..3]
        .iter()
        .all(|stride| stride % stride_align == 0);

    let valid_layout = match ident {
        InputIdent::Lhs => handle.strides[3] == 1,
        InputIdent::Rhs => {
            let c_major = handle.strides[3] == 1;
            let kernel_contig = handle.strides[2] * handle.shape[2] == handle.strides[1];
            c_major && kernel_contig
        }
    };

    valid_layout && aligned
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
