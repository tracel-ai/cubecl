use std::any::TypeId;

use cubecl_core::{Runtime, client::ComputeClient, prelude::*};
use half::f16;

use crate::matmul::{
    components::{
        MatmulLineSizes,
        global::args::{ConcreteOutputFactory, MatmulArgs},
    },
    kernels::MatmulLaunchError,
};
use crate::{
    convolution::base::ConvolutionLaunch,
    matmul::components::{self, InputIdent, MatmulPrecision},
};

use super::{
    ConvLaunchError,
    algorithm::{Algorithm, StageInput},
    args::ConvInputsLaunch,
    base::{ConvolutionProblem, Dimensionality},
    selection::select_matmul,
};

type Input<Alg, MP> = <<Alg as Algorithm>::Args as MatmulArgs>::Input<<MP as MatmulPrecision>::EI>;
type Output<Alg, MP> =
    <<Alg as Algorithm>::Args as MatmulArgs>::Output<<MP as MatmulPrecision>::EO>;

#[derive(Clone)]
pub struct ConvolutionArgs<const N_SPATIAL: usize> {
    pub stride: [usize; N_SPATIAL],
    pub padding: [usize; N_SPATIAL],
    pub dilation: [usize; N_SPATIAL],
}

/// Perform an n-dimensional convolution using the implicit GEMM (im2col) algorithm, using cubecl
/// tiling matmul components, using the specified algorithm.
///
/// * `input` - The input feature map, layout should be [batches, depth, height, width, in_channels]
/// * `weight` - The weights (filter) applied to each kernel, layout should be [out_channels, kernel_d, kernel_h, kernel_w, in_channels]
/// * `out` - The output feature map, layout should be [batches, out_depth, out_height, out_width, out_channels]
/// * `bias` - The bias added to each out channel
/// * `options` - The options to use for the convolution
#[allow(clippy::result_large_err)]
pub fn launch_conv<R: Runtime, MP: MatmulPrecision, Alg: Algorithm, const N_SPATIAL: usize>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    weight: &TensorHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs<N_SPATIAL>,
) -> Result<(), ConvLaunchError>
where
    Input<Alg, MP>: ConvInputsLaunch,
    Output<Alg, MP>: ConcreteOutputFactory,
{
    let ConvolutionArgs {
        stride,
        padding,
        dilation,
    } = args;

    let dimensionality = match N_SPATIAL {
        1 => Dimensionality::Dim1,
        2 => Dimensionality::Dim2,
        3 => Dimensionality::Dim3,
        other => unimplemented!("Unsupported dimensionality {other}"),
    };

    launch::<R, MP, Alg>(
        client,
        input,
        weight,
        bias,
        out,
        (&stride, &padding, &dilation),
        dimensionality,
    )
}

fn launch<R: Runtime, MP: MatmulPrecision, Alg: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    weight: &TensorHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
) -> Result<(), ConvLaunchError>
where
    Input<Alg, MP>: ConvInputsLaunch,
    Output<Alg, MP>: ConcreteOutputFactory,
{
    let rank = input.shape.len();
    let dim_c = rank - 1;

    let n = input.shape[0];
    let c = input.shape[dim_c];

    let out_c = weight.shape[0];

    let in_shape = &input.shape[1..dim_c];
    let kernel_shape = &weight.shape[1..dim_c];
    let out_shape = &out.shape[1..dim_c];

    let input = Alg::into_tensor_handle::<R, MP::EI>(client, input, InputIdent::Lhs);
    let weight = Alg::into_tensor_handle::<R, MP::EI>(client, weight, InputIdent::Rhs);

    let plane_dim = client
        .properties()
        .hardware
        .defined_plane_size()
        .unwrap_or(32);

    let problem = ConvolutionProblem {
        m: n * out_shape.iter().product::<usize>(),
        n: out_c,
        k: c * kernel_shape.iter().product::<usize>(),
        lhs_layout: components::MatrixLayout::RowMajor,
        rhs_layout: components::MatrixLayout::ColMajor,
        kernel_size: kernel_shape.iter().map(|it| *it as u32).collect(),
        stride: stride.iter().map(|it| *it as u32).collect(),
        padding: padding.iter().map(|it| *it as i32).collect(),
        dilation: dilation.iter().map(|it| *it as u32).collect(),

        batches: n,
        shape: in_shape.to_vec(),
        out_shape: out_shape.to_vec(),
        channels: c,

        dimensionality,
    };

    let line_sizes = Alg::line_sizes(
        &problem,
        R::line_size_elem(&MP::EI::as_elem_native_unchecked()),
        R::line_size_elem(&MP::EO::as_elem_native_unchecked()),
    );

    let (selection, config_input) = select_matmul::<Alg, R>(
        client,
        &problem,
        plane_dim,
        MP::ES::as_elem_native_unchecked(),
        MP::EA::as_elem_native_unchecked(),
    );

    let launch = if TypeId::of::<MP::EI>() == TypeId::of::<f32>() {
        if tf32::is_supported(client) {
            launch_kernel::<R, (MP::EI, tf32, f32, MP::EO), Alg>
        } else {
            launch_kernel::<R, (MP::EI, f16, f32, MP::EO), Alg>
        }
    } else {
        launch_kernel::<R, MP, Alg>
    };

    launch(
        client,
        &input.as_ref(),
        &weight.as_ref(),
        bias,
        out,
        problem,
        &line_sizes,
        selection,
        config_input,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, MP: MatmulPrecision, Alg: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    weight: &TensorHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    line_sizes: &MatmulLineSizes,
    selection: Alg::MatmulSelection,
    config_input: StageInput,
) -> Result<(), ConvLaunchError>
where
    Input<Alg, MP>: ConvInputsLaunch,
    Output<Alg, MP>: ConcreteOutputFactory,
{
    let rank = out.shape.len();
    let dim_c = rank - 1;

    // Reshape out to (M, N)
    let out_shape = [out.shape[0..dim_c].iter().product(), out.shape[dim_c]];
    let out_strides = [out.strides[rank - 2], out.strides[dim_c]];

    let out = unsafe {
        TensorHandleRef::from_raw_parts(out.handle, &out_strides, &out_shape, out.elem_size)
    };

    let cube_dim = Alg::cube_dim(&selection);
    let cube_count = Alg::cube_count(&selection, &problem);

    let config = Alg::make_config::<R, MP>(
        client,
        (config_input, Alg::loading_precompute_strategy()),
        &problem,
        line_sizes,
        &cube_dim,
        &cube_count,
    )
    .map_err(MatmulLaunchError::InvalidConfig)?;

    Alg::check_availability::<R, MP>(client, &config)?;

    let input = <Input<Alg, MP> as ConvInputsLaunch>::create(
        input, weight, &selection, &problem, line_sizes,
    );
    let output = <Output<Alg, MP> as ConcreteOutputFactory>::create(
        &out,
        &selection,
        &problem.as_matmul_problem(),
        line_sizes,
    );
    let bias = bias.as_ref().map(|bias| bias.as_tensor_arg(line_sizes.out));

    unsafe {
        Alg::GlobalConvolution::launch_unchecked::<(MP, Alg::Args), R>(
            client, cube_dim, cube_count, input, bias, output, &problem, config,
        );
    }

    Ok(())
}
