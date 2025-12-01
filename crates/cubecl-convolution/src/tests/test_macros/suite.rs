use crate::{
    components::{
        ConvolutionProblem, Dimensionality,
        global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
    },
    tests::test_utils::TestPrecision,
};
use crate::{
    kernels::layered::algorithm::Algorithm,
    tests::convolution_test_launcher::test_convolution_algorithm,
};
use cubecl_core::Runtime;
use cubecl_matmul::components::{InputArg, OutputArg};
use cubecl_matmul::components::{MatmulSelection, MatrixLayout};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionSize {
    pub h: usize,
    pub w: usize,
    pub c: usize,

    pub out_c: usize,
}

pub fn test_algo<A: Algorithm, P: TestPrecision, R: Runtime>(
    selection: MatmulSelection,
    problem: ConvolutionSize,
) where
    InputArg<A::Args>: ConcreteInputsFactory,
    OutputArg<A::Args>: ConcreteOutputFactory,
{
    let client = R::client(&Default::default());

    // TODO: Automate more params
    let batches = 2;
    let kernel_size = vec![4, 3];
    let stride = vec![1, 1];
    let padding = vec![3, 1];
    let dilation = vec![3, 2];

    let out_h = calculate_conv_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        problem.h,
    );
    let out_w = calculate_conv_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        problem.w,
    );

    let problem = ConvolutionProblem {
        m: batches * out_h * out_w,
        n: problem.out_c,
        k: kernel_size.iter().product::<u32>() as usize * problem.c,
        lhs_strides: vec![],
        rhs_strides: vec![],
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        kernel_size,
        stride,
        padding,
        dilation,
        batches,
        shape: vec![problem.h, problem.w],
        channels: problem.c,
        out_shape: vec![out_h, out_w],
        dimensionality: Dimensionality::Dim2,
    };

    test_convolution_algorithm::<A, P, R>(client, problem, selection);
}

/// Calculate the expected output size when doing a convolution operation.
pub fn calculate_conv_output_size(
    kernel_size: u32,
    stride: u32,
    padding: i32,
    dilation: u32,
    size_in: usize,
) -> usize {
    (size_in + 2 * padding as usize - dilation as usize * (kernel_size as usize - 1) - 1)
        / stride as usize
        + 1
}
