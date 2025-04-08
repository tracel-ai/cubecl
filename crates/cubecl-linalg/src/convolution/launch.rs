use std::any::TypeId;

use cubecl_core::{Runtime, client::ComputeClient, prelude::*, tensor_line_size_parallel};
use half::f16;

use crate::{
    convolution::base::ConvolutionLaunch,
    matmul::components::{self, MatmulPrecision, MatmulSelection},
};
use crate::{
    matmul::{
        components::global::args::{MatmulArgs, OutputLaunch},
        kernels::MatmulLaunchError,
    },
    tensor::into_contiguous_pitched,
};

use super::{
    ConvLaunchError,
    algorithm::{Algorithm, StageInput},
    args::ConvInputsLaunch,
    base::{ConvolutionConfigFactory, ConvolutionProblem},
    selection::select_matmul,
};

type Input<Alg, MP> = <<Alg as Algorithm>::Args as MatmulArgs>::Input<<MP as MatmulPrecision>::EI>;
type Output<Alg, MP> =
    <<Alg as Algorithm>::Args as MatmulArgs>::Output<<MP as MatmulPrecision>::EO>;

pub struct ConvolutionArgs {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
#[allow(clippy::result_large_err)]
pub fn launch_conv2d_nhwc<R: Runtime, MP: MatmulPrecision, Alg: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    weight: &TensorHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs,
) -> Result<(), ConvLaunchError>
where
    Input<Alg, MP>: ConvInputsLaunch,
{
    let in_valid = Alg::has_valid_layout(input);
    let weight_valid = Alg::has_valid_layout(weight);

    match (in_valid, weight_valid) {
        (true, true) => prepare_problem::<R, MP, Alg>(client, input, weight, bias, out, args),
        (true, false) => {
            let weight = into_contiguous_pitched::<R, MP::EI>(client, weight);
            prepare_problem::<R, MP, Alg>(client, input, &weight.as_ref(), bias, out, args)
        }
        (false, true) => {
            let input = into_contiguous_pitched::<R, MP::EI>(client, input);
            prepare_problem::<R, MP, Alg>(client, &input.as_ref(), weight, bias, out, args)
        }
        (false, false) => {
            let input = into_contiguous_pitched::<R, MP::EI>(client, input);
            let weight = into_contiguous_pitched::<R, MP::EI>(client, weight);
            prepare_problem::<R, MP, Alg>(
                client,
                &input.as_ref(),
                &weight.as_ref(),
                bias,
                out,
                args,
            )
        }
    }
}

fn prepare_problem<R: Runtime, MP: MatmulPrecision, Alg: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    weight: &TensorHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs,
) -> Result<(), ConvLaunchError>
where
    Input<Alg, MP>: ConvInputsLaunch,
{
    let ConvolutionArgs {
        stride,
        padding,
        dilation,
    } = args;

    let [n, h, w, c] = input.shape.try_into().unwrap();
    let [kh, kw, _, out_c] = weight.shape.try_into().unwrap();
    let out_h = out.shape[1];
    let out_w = out.shape[2];

    let ei_elem = MP::EI::as_elem_native_unchecked();
    let eo_elem = MP::EO::as_elem_native_unchecked();

    let lhs_line_size =
        tensor_line_size_parallel(R::line_size_elem(&ei_elem), input.shape, input.strides, 3);
    let rhs_line_size =
        tensor_line_size_parallel(R::line_size_elem(&ei_elem), weight.shape, weight.strides, 3);

    let out_line_size =
        tensor_line_size_parallel(R::line_size_elem(&eo_elem), out.shape, out.strides, 3);

    let plane_dim = client
        .properties()
        .hardware_properties()
        .defined_plane_size()
        .unwrap_or(32);

    let problem = ConvolutionProblem {
        m: n * out_h * out_w,
        n: out_c,
        k: c * kh * kw,
        lhs_layout: components::MatrixLayout::RowMajor,
        rhs_layout: components::MatrixLayout::RowMajor,
        lhs_line_size,
        rhs_line_size,
        out_line_size,
        kernel_size: (kh as u32, kw as u32),
        stride: (stride.0 as u32, stride.1 as u32),
        padding: (padding.0 as i32, padding.1 as i32),
        dilation: (dilation.0 as u32, dilation.1 as u32),

        batches: n,
        height: h,
        width: w,
        channels: c,

        out_h,
        out_w,
    };

    let (selection, config_input) = select_matmul::<Alg, R, MP>(client, &problem, plane_dim);

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
        input,
        weight,
        bias,
        out,
        problem,
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
    selection: MatmulSelection,
    config_input: StageInput,
) -> Result<(), ConvLaunchError>
where
    Input<Alg, MP>: ConvInputsLaunch,
{
    // Reshape weight to (K, N)
    let weight_shape = [weight.shape[0..3].iter().product(), weight.shape[3]];
    let weight_strides = [weight.strides[2], weight.strides[3]];

    let weight = unsafe {
        TensorHandleRef::from_raw_parts(
            weight.handle,
            &weight_strides,
            &weight_shape,
            weight.elem_size,
        )
    };

    // Reshape out to (M, N)
    let out_shape = [out.shape[0..3].iter().product(), out.shape[3]];
    let out_strides = [out.strides[2], out.strides[3]];

    let out = unsafe {
        TensorHandleRef::from_raw_parts(out.handle, &out_strides, &out_shape, out.elem_size)
    };

    let cube_dim = Alg::cube_dim(&selection);
    let cube_count = Alg::cube_count(&selection, &problem);

    let config = Alg::make_config(config_input, &problem, &cube_dim, &cube_count)
        .map_err(MatmulLaunchError::InvalidConfig)?;

    <Alg::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, MP>(
        client, &config,
    )?;

    let input = <Input<Alg, MP> as ConvInputsLaunch>::create(input, &weight, &selection, &problem);
    let output =
        <Output<Alg, MP> as OutputLaunch>::create(&out, &selection, &problem.as_matmul_problem());
    let bias = bias
        .as_ref()
        .map(|bias| bias.as_tensor_arg(problem.out_line_size));

    unsafe {
        Alg::GlobalConvolution::launch_unchecked::<(MP, Alg::Args), R>(
            client, cube_dim, cube_count, input, bias, output, &problem, config,
        );
    }

    Ok(())
}
