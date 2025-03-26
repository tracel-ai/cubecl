use cubecl_core::{
    Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorArg},
};

use crate::matmul::kernels::MatmulLaunchError;
use crate::{convolution::base::ConvolutionLaunch, matmul::components::MatmulPrecision};

use super::{
    ConvLaunchError,
    algorithm::Algorithm,
    base::{ConvolutionConfigFactory, ConvolutionProblem},
    selection::ConvSelector,
};

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
#[allow(clippy::result_large_err)]
pub fn launch_conv2d_nhwc<R: Runtime, SP: MatmulPrecision, Alg: Algorithm, S: ConvSelector<Alg>>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorArg<R>,
    weight: TensorArg<R>,
    bias: TensorArg<R>,
    out: TensorArg<R>,
    problem: ConvolutionProblem,
) -> Result<(), ConvLaunchError>
where
    SP::EG: Numeric,
{
    let plane_dim = client
        .properties()
        .hardware_properties()
        .defined_plane_size()
        .unwrap_or(32);

    let (selection, config_input) = S::select_kernel::<R, SP>(client, &problem, plane_dim);
    let cube_dim = Alg::cube_dim(&selection);
    let cube_count = Alg::cube_count(&selection, &problem);

    let config = Alg::make_config(config_input, &problem, &cube_dim, &cube_count)
        .map_err(MatmulLaunchError::InvalidConfig)?;

    <Alg::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, SP>(
        client, &config,
    )?;

    unsafe {
        Alg::GlobalConvolution::launch_unchecked::<SP, R>(
            client, cube_dim, cube_count, input, weight, bias, out, config,
        );
    }

    Ok(())
}
