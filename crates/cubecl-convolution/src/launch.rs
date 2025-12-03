use crate::components::ConvGemmConfig as _;
use crate::{components::ConvSetupError, kernels::layered::selector::launch_kernel_concrete};
use crate::{
    components::{
        ConvolutionProblem, Dimensionality,
        global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
    },
    kernels::layered::algorithm::Algorithm,
};
use cubecl_core::{Runtime, client::ComputeClient, prelude::*};
use cubecl_matmul::components::{self, AvailableLineSizes, MatmulElems, MatrixLayout};
use cubecl_matmul::{
    MatmulInputHandleRef,
    components::{InputArg, OutputArg},
};

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
pub fn launch_conv<R: Runtime, Alg: Algorithm, const N_SPATIAL: usize>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    InputArg<Alg::Args>: ConcreteInputsFactory,
    OutputArg<Alg::Args>: ConcreteOutputFactory,
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

    launch::<R, Alg>(
        client,
        input,
        weight,
        bias,
        out,
        (&stride, &padding, &dilation),
        dimensionality,
        dtypes,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    (stride, padding, dilation): (&[usize], &[usize], &[usize]),
    dimensionality: Dimensionality,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    InputArg<Alg::Args>: ConcreteInputsFactory,
    OutputArg<Alg::Args>: ConcreteOutputFactory,
{
    let rank = input.data().shape.len();
    let dim_c = rank - 1;

    let n = input.data().shape[0];
    let c = input.data().shape[dim_c];

    let out_c = weight.data().shape[0];

    let in_shape = &input.data().shape[1..dim_c];
    let kernel_shape = &weight.data().shape[1..dim_c];
    let out_shape = &out.shape[1..dim_c];

    let input_data = Alg::into_tensor_handle(client, input.data(), *dtypes.lhs_global)?;
    let weight_data = Alg::into_tensor_handle(client, weight.data(), *dtypes.rhs_global)?;

    let mut input = *input;
    let mut weight = *weight;

    *input.data_mut() = input_data.as_ref();
    *weight.data_mut() = weight_data.as_ref();

    let problem = ConvolutionProblem {
        m: n * out_shape.iter().product::<usize>(),
        n: out_c,
        k: c * kernel_shape.iter().product::<usize>(),
        lhs_strides: input.data().strides.to_vec(),
        rhs_strides: weight.data().strides.to_vec(),
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

    launch_kernel::<R, Alg>(client, &input, &weight, bias, out, problem, dtypes)
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel<R: Runtime, Alg: Algorithm>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    mut dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    InputArg<Alg::Args>: ConcreteInputsFactory,
    OutputArg<Alg::Args>: ConcreteOutputFactory,
{
    let plane_dim = client.properties().hardware.plane_size_max;
    // Shape/strides are treated as k-major, with the last dim always being the contiguous one.
    // So for the sake of selecting a line size, the shape/strides are always row-major.
    let line_sizes = AvailableLineSizes::from_type_sizes(
        client,
        input.data().elem_size,
        weight.data().elem_size,
        out.elem_size,
    )
    .filter_lhs_with_tensor(
        input.data().strides,
        input.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_rhs_with_tensor(
        weight.data().strides,
        weight.data().shape,
        MatrixLayout::RowMajor,
    )
    .filter_out_with_tensor(out.strides, out.shape);

    let line_sizes = Alg::filter_line_sizes(line_sizes).pick_max()?;

    let selection = Alg::selection(client, &problem, plane_dim, &line_sizes, &mut dtypes)?;

    let config = Alg::setup(client, &problem, &selection, &line_sizes, &dtypes)?;

    let line_sizes = config.line_sizes();

    launch_kernel_concrete::<R, Alg>(
        client, input, weight, bias, out, problem, line_sizes, selection, &dtypes,
    )
}
