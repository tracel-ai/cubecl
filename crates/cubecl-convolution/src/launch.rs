use std::any::TypeId;

use cubecl_core::{Runtime, client::ComputeClient, prelude::*};
use cubecl_matmul::MatmulInputHandleRef;
use half::f16;

use crate::{
    components::{ConvGemmConfig as _, ConvSetupError},
    kernels::layered::selector::launch_kernel_concrete,
};
use crate::{
    components::{ConvolutionProblem, Dimensionality, global::args::ConcreteInputsFactory},
    kernels::layered::algorithm::Algorithm,
};
use cubecl_matmul::components::global::args::{ConcreteOutputFactory, MatmulArgs};
use cubecl_matmul::components::{
    self, AvailableLineSizes, InputPrecision, LhsG, MatmulElems, MatmulIdent, MatmulPrecision,
    MatmulSelection, RhsG,
};

type Input<Alg, MP> = <<Alg as Algorithm>::Args as MatmulArgs>::Input<
    <<MP as MatmulPrecision>::Lhs as InputPrecision>::Global,
    <<MP as MatmulPrecision>::Rhs as InputPrecision>::Global,
    <MP as MatmulPrecision>::EO,
>;
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
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs<N_SPATIAL>,
) -> Result<(), ConvSetupError>
where
    Input<Alg, MP>: ConcreteInputsFactory,
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
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
) -> Result<(), ConvSetupError>
where
    Input<Alg, MP>: ConcreteInputsFactory,
    Output<Alg, MP>: ConcreteOutputFactory,
{
    let rank = input.data().shape.len();
    let dim_c = rank - 1;

    let n = input.data().shape[0];
    let c = input.data().shape[dim_c];

    let out_c = weight.data().shape[0];

    let in_shape = &input.data().shape[1..dim_c];
    let kernel_shape = &weight.data().shape[1..dim_c];
    let out_shape = &out.shape[1..dim_c];

    let input_data = Alg::into_tensor_handle::<R, LhsG<MP>>(client, input.data(), MatmulIdent::Lhs);
    let weight_data =
        Alg::into_tensor_handle::<R, RhsG<MP>>(client, weight.data(), MatmulIdent::Rhs);

    let mut input = *input;
    let mut weight = *weight;

    *input.data_mut() = input_data.as_ref();
    *weight.data_mut() = weight_data.as_ref();

    let plane_dim = client.properties().hardware.plane_size_max;

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

    let selection = Alg::selection::<R>(client, &problem, plane_dim, MatmulElems::new::<MP>())?;

    let lhs_is_f32 = TypeId::of::<LhsG<MP>>() == TypeId::of::<f32>();
    let rhs_is_f32 = TypeId::of::<RhsG<MP>>() == TypeId::of::<f32>();

    let launch = if lhs_is_f32 || rhs_is_f32 {
        if tf32::is_supported(client) {
            if lhs_is_f32 && rhs_is_f32 {
                launch_kernel::<R, (LhsG<MP>, RhsG<MP>, tf32, tf32, f32, AccG<MP>), Alg>
            } else if lhs_is_f32 {
                launch_kernel::<R, (LhsG<MP>, RhsG<MP>, tf32, f32, f32, AccG<MP>), Alg>
            } else {
                launch_kernel::<R, (LhsG<MP>, RhsG<MP>, f32, tf32, f32, AccG<MP>), Alg>
            }
        } else if lhs_is_f32 && rhs_is_f32 {
            launch_kernel::<R, (LhsG<MP>, RhsG<MP>, f16, f16, f32, AccG<MP>), Alg>
        } else if lhs_is_f32 {
            launch_kernel::<R, (LhsG<MP>, RhsG<MP>, f16, f32, f32, AccG<MP>), Alg>
        } else {
            launch_kernel::<R, (LhsG<MP>, RhsG<MP>, f32, f16, f32, AccG<MP>), Alg>
        }
    } else {
        launch_kernel::<R, MP, Alg>
    };

    launch(client, &input, &weight, bias, out, problem, selection)
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, MP: MatmulPrecision, Alg: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    selection: MatmulSelection,
) -> Result<(), ConvSetupError>
where
    Input<Alg, MP>: ConcreteInputsFactory,
    Output<Alg, MP>: ConcreteOutputFactory,
{
    let line_sizes = AvailableLineSizes::from_types::<R>(
        &LhsG::<MP>::as_type_native_unchecked(),
        &RhsG::<MP>::as_type_native_unchecked(),
        &AccG<MP>::as_type_native_unchecked(),
    )
    .filter_lhs_with_tensor(input.data().strides, input.data().shape, problem.lhs_layout)
    .filter_rhs_with_tensor(
        weight.data().strides,
        weight.data().shape,
        problem.rhs_layout,
    )
    .filter_out_with_tensor(out.strides, out.shape);

    let line_sizes = Alg::filter_line_sizes(line_sizes).pick_max()?;

    let config = Alg::setup::<R, MP>(client, &problem, &selection, &line_sizes)?;

    let line_sizes = config.line_sizes();

    launch_kernel_concrete::<(MP, Alg::Args), R, Alg>(
        client, input, weight, bias, out, problem, line_sizes, selection,
    )
}
