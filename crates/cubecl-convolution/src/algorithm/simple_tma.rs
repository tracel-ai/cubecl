use std::marker::PhantomData;

use cubecl_core::ir::Elem;
use cubecl_core::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
};

use crate::{
    base::{ConvolutionConfigFactory, ConvolutionProblem, Dimensionality},
    homogeneous::simple_tma::SimpleTmaConvolutionFamily,
    selection::convolution_matmul_selection,
};
use cubecl_matmul::components::MatmulLineSizes;
use cubecl_matmul::components::stage::NumStages;
use cubecl_matmul::kernels::matmul::MatmulSelection;
use cubecl_matmul::{
    components::{
        InputIdent, InvalidConfigError, MatmulPrecision,
        global::args::TensorMapArgs,
        stage::{FullReaderFamily, plane_matmul::PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    kernels::matmul::PlaneMatmulSelection,
};

use cubecl_std::tensor::{TensorHandle, into_contiguous_pitched};

use super::Algorithm;

pub const TMA_STRIDE_ALIGN: usize = 16;

/// Cmma convolution
pub struct SimpleTmaConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for SimpleTmaConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalConvolution = SimpleTmaConvolutionFamily<Self::StageMatmul>;
    type MatmulSelection = PlaneMatmulSelection;

    type Args = TensorMapArgs;

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        CubeDim::new(
            selection.plane_dim,
            selection.tiling_scheme().tiles_in_stage_m(),
            1,
        )
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme().elements_in_stage_m();
        let n_stage = selection.tiling_scheme().elements_in_stage_n();
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn make_config<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        check_problem_tma(problem)?;

        let config = Self::GlobalConvolution::make_config::<R, MP>(
            client, input, problem, line_sizes, cube_dim, cube_count,
        );
        Self::GlobalConvolution::check_config(&config)?;
        Ok(config)
    }

    fn check_availability<R: Runtime, MP: cubecl_matmul::components::MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::GlobalConvolution as ConvolutionConfigFactory>::Config,
    ) -> Result<(), cubecl_matmul::kernels::MatmulAvailabilityError> {
        <Self::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, MP>(
            client, config,
        )?;

        if !client
            .properties()
            .feature_enabled(cubecl_core::Feature::Tma(cubecl_core::TmaFeature::Base))
        {
            return Err(cubecl_matmul::kernels::MatmulAvailabilityError::TmaUnavailable);
        }

        Ok(())
    }

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: InputIdent,
    ) -> TensorHandle<R, E> {
        into_tensor_handle_tma(client, handle, ident)
    }

    // TODO this is not the same as tma stages, it's stages in the sense of double buffering in matmul
    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> Self::MatmulSelection {
        convolution_matmul_selection::<TMM, R>(client, problem, plane_dim, elem_stage, elem_acc)
    }
}

pub(crate) fn into_tensor_handle_tma<R: Runtime, E: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    handle: &TensorHandleRef<'_, R>,
    ident: InputIdent,
) -> TensorHandle<R, E> {
    let rank = handle.shape.len();
    let dim_c = rank - 1;
    let mut handle = if has_valid_layout(handle, ident) {
        TensorHandle::from_ref(handle)
    } else {
        into_contiguous_pitched(client, handle)
    };
    match ident {
        InputIdent::Lhs => handle,
        InputIdent::Rhs => {
            let k_size = handle.shape[1..dim_c].iter().product();
            handle.shape = vec![handle.shape[0], k_size, handle.shape[dim_c]];
            handle.strides = vec![
                handle.strides[0],
                handle.strides[dim_c - 1],
                handle.strides[dim_c],
            ];
            handle
        }
    }
}

pub(crate) fn has_valid_layout<R: Runtime>(
    handle: &TensorHandleRef<'_, R>,
    ident: InputIdent,
) -> bool {
    let stride_align = TMA_STRIDE_ALIGN / handle.elem_size;
    let rank = handle.shape.len();
    let dim_c = rank - 1;

    let aligned = handle.strides[..dim_c]
        .iter()
        .all(|stride| stride % stride_align == 0);

    let valid_layout = match ident {
        InputIdent::Lhs => handle.strides[dim_c] == 1,
        InputIdent::Rhs => {
            let c_major = handle.strides[dim_c] == 1;
            let mut kernel_contig = true;
            for i in 1..dim_c - 1 {
                kernel_contig &= handle.strides[i] == handle.strides[i + 1] * handle.shape[i + 1];
            }
            c_major && kernel_contig
        }
    };

    valid_layout && aligned
}

pub(crate) fn check_problem_tma(problem: &ConvolutionProblem) -> Result<(), InvalidConfigError> {
    fn check_range(
        value: isize,
        name: impl FnOnce() -> String,
        min: isize,
        max: isize,
    ) -> Result<(), InvalidConfigError> {
        if value < min || value > max {
            let name = name();
            Err(Box::new(format!(
                "value {name} outside of valid range ({min}, {max})"
            )))
        } else {
            Ok(())
        }
    }

    let (corner_min, corner_max) = match problem.dimensionality {
        Dimensionality::Dim1 => (-(2isize.pow(15)), 2isize.pow(15) - 1),
        Dimensionality::Dim2 => (-(2isize.pow(7)), 2isize.pow(7) - 1),
        Dimensionality::Dim3 => (-(2isize.pow(4)), 2isize.pow(4) - 1),
    };

    let corner = calculate_upper_corner(&problem.padding, &problem.kernel_size, &problem.dilation);
    for (i, offs) in corner.iter().enumerate() {
        check_range(
            *offs as isize,
            || format!("corner[{i}]"),
            corner_min,
            corner_max,
        )?;
    }

    let offset_max = match problem.dimensionality {
        Dimensionality::Dim1 => 2isize.pow(16) - 1,
        Dimensionality::Dim2 => 2isize.pow(8) - 1,
        Dimensionality::Dim3 => 2isize.pow(5) - 1,
    };

    for i in 0..problem.kernel_size.len() {
        let offset = (problem.kernel_size[i] - 1) * problem.dilation[i];
        check_range(
            offset as isize,
            || format!("kernel size {i}"),
            0,
            offset_max,
        )?;
        check_range(problem.stride[i] as isize, || format!("stride[{i}]"), 1, 8)?;
    }

    Ok(())
}

pub fn calculate_lower_corner(padding: &[i32]) -> Vec<i32> {
    padding.iter().map(|padding| -*padding).collect()
}

pub fn calculate_upper_corner(padding: &[i32], kernel_size: &[u32], dilation: &[u32]) -> Vec<i32> {
    padding
        .iter()
        .zip(kernel_size)
        .zip(dilation)
        .map(|((padding, kernel_size), dilation)| {
            *padding - (*kernel_size - 1) as i32 * *dilation as i32
        })
        .collect()
}
