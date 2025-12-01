use cubecl_core::server::LaunchError;
use cubecl_core::{
    CubeCount, Runtime, client::ComputeClient, ir::StorageType, prelude::TensorHandleRef,
};
use cubecl_matmul::components::{
    InvalidConfigError, global::args::TensorMapArgs, stage::PlaneMatmulFamily,
    tile::TileMatmulFamily,
};
use cubecl_matmul::components::{
    MatmulElems, MatmulSelection, MatmulSetupError, stage::StridedStageFamily, tile::io::Strided,
};
use cubecl_matmul::components::{MatmulLineSizes, stage::NumStages};
use cubecl_std::{
    CubeOption,
    tensor::{TensorHandle, into_contiguous_pitched},
};
use std::marker::PhantomData;

use crate::components::{
    ConvolutionProblem, Dimensionality, convolution_matmul_selection,
    global::single_stage::tma::SimpleTmaConvolutionFamily,
};

use super::Algorithm;

pub const TMA_STRIDE_ALIGN: usize = 16;

/// Cmma convolution
pub struct SimpleTmaConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<
    TMM: TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
> Algorithm for SimpleTmaConvAlgorithm<TMM>
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        Option<StridedStageFamily>,
    >;
    type GlobalConvolution = SimpleTmaConvolutionFamily<Self::StageMatmul>;

    type Args = TensorMapArgs;

    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme.elements_per_stage_along_m();
        let n_stage = selection.tiling_scheme.elements_per_stage_along_n();
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TensorHandleRef<'_, R>,
        dtype: StorageType,
    ) -> Result<TensorHandle<R>, LaunchError> {
        into_tensor_handle_tma(client, handle, dtype)
    }

    // TODO this is not the same as tma stages, it's stages in the sense of double buffering in matmul
    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(convolution_matmul_selection::<TMM, R>(
            client, problem, plane_dim, false, line_sizes, dtypes,
        )?)
    }
}

pub(crate) fn into_tensor_handle_tma<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<TensorHandle<R>, LaunchError> {
    let handle = if has_valid_layout(handle) {
        TensorHandle::from_ref(handle, dtype)
    } else {
        into_contiguous_pitched(client, handle, dtype)?
    };
    Ok(handle)
}

pub(crate) fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>) -> bool {
    let stride_align = TMA_STRIDE_ALIGN / handle.elem_size;
    let rank = handle.shape.len();
    let dim_c = rank - 1;

    let aligned = handle.strides[..dim_c]
        .iter()
        .all(|stride| stride % stride_align == 0);

    let valid_layout = handle.strides[dim_c] == 1;

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
