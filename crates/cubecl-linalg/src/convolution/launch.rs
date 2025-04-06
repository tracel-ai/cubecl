use cubecl_core::{Runtime, client::ComputeClient, prelude::*, tensor_line_size_parallel};

use crate::{
    convolution::base::ConvolutionLaunch,
    matmul::components::{self, MatmulPrecision, MatmulSelection},
};
use crate::{
    matmul::{
        components::{
            EI, EO, MatmulSpec,
            global::args::{MatmulArgs, OutputLaunch},
        },
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

pub struct ConvolutionArgs {
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
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
) -> Result<(), ConvLaunchError> {
    let in_contiguous = is_contiguous_4d(input);
    let weight_contiguous = is_contiguous_4d(weight);

    match (in_contiguous, weight_contiguous) {
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
) -> Result<(), ConvLaunchError> {
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

    let mut problem = ConvolutionProblem {
        m: n * h * w,
        n: out_c,
        k: c * kh * kw,
        lhs_layout: components::MatrixLayout::RowMajor,
        rhs_layout: components::MatrixLayout::RowMajor,
        lhs_line_size,
        rhs_line_size,
        out_line_size,
        padded_channels: 0,
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

    // Pad channels to tile size to align loads, so we only ever load one kernel position at a time
    // This may improve memory coalescing, and is necessary for TMA to work properly
    // In practice, this is usually already aligned anyways, since channels tend to be a power of
    // two.
    problem.padded_channels = (c as u32).next_multiple_of(selection.tile_shape.k);

    launch_kernel::<R, MP, Alg>(
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

fn is_contiguous_4d<R: Runtime>(handle: &TensorHandleRef<'_, R>) -> bool {
    let mut strides = handle.strides.to_vec();
    strides.sort();

    // Permuted strides
    if handle.strides != strides {
        return false;
    }

    // channels doesn't need to be contiguous with the rest of the tensor
    strides[2] * handle.shape[2] == strides[1] && strides[1] * handle.shape[1] == handle.strides[0]
}

type Input<MS> = <<MS as MatmulSpec>::Args as MatmulArgs>::Input<EI<MS>>;
type Output<MS> = <<MS as MatmulSpec>::Args as MatmulArgs>::Output<EO<MS>>;

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, MS: MatmulSpec, Alg: Algorithm>(
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
    Input<MS>: ConvInputsLaunch,
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

    <Alg::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, MS::Precision>(
        client, &config,
    )?;

    let input = <Input<MS> as ConvInputsLaunch>::create(input, &weight, &selection, &problem);
    let output =
        <Output<MS> as OutputLaunch>::create(&out, &selection, &problem.as_matmul_problem());
    let bias = bias
        .as_ref()
        .map(|bias| bias.as_tensor_arg(problem.out_line_size));

    unsafe {
        Alg::GlobalConvolution::launch_unchecked::<MS, R>(
            client, cube_dim, cube_count, input, bias, output, config,
        );
    }

    Ok(())
}
