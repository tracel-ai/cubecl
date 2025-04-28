use crate::convolution::{
    algorithm::Algorithm, args::ConvInputsLaunch, tests::test_utils::TestPrecision,
};
use crate::matmul::components::stage::StageVectorization;
use crate::matmul::components::{CompleteStageTiling, MatrixLayout};
use crate::{
    convolution::base::ConvolutionProblem, matmul::components::global::args::ConcreteOutputFactory,
};
use crate::{
    convolution::tests::convolution_test_launcher::test_convolution_algorithm,
    matmul::components::{
        MatmulSelection, MatmulSize, global::args::MatmulArgs, stage::STAGE_BUFFERING,
    },
};
use cubecl_core::Runtime;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionSize {
    pub h: usize,
    pub w: usize,
    pub c: usize,

    pub out_c: usize,
}

pub fn test_algo<A: Algorithm, Args: MatmulArgs, P: TestPrecision, R: Runtime>(
    tile_shape: MatmulSize,
    tile_count: MatmulSize,
    problem: ConvolutionSize,
) where
    Args::Input<P::EG>: ConvInputsLaunch,
    Args::Output<P::EG>: ConcreteOutputFactory,
{
    let client = R::client(&Default::default());
    let plane_dim = match client
        .properties()
        .hardware_properties()
        .defined_plane_size()
    {
        Some(val) => val,
        None => {
            println!("Can't run test without a fixed plane size.");
            return;
        }
    };

    // TODO: Automate more params
    let batches = 2;
    let kernel_size = (4, 3);
    let stride = (1, 1);
    let padding = (3, 1);
    let dilation = (3, 2);

    let out_h =
        calculate_conv_output_size(kernel_size.0, stride.0, padding.0, dilation.0, problem.h);
    let out_w =
        calculate_conv_output_size(kernel_size.1, stride.1, padding.1, dilation.1, problem.w);

    let problem = ConvolutionProblem {
        m: batches * out_h * out_w,
        n: problem.out_c,
        k: kernel_size.0 as usize * kernel_size.1 as usize * problem.c,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        lhs_line_size: 1,
        rhs_line_size: 1,
        out_line_size: 1,
        kernel_size,
        stride,
        padding,
        dilation,
        batches,
        height: problem.h,
        width: problem.w,
        channels: problem.c,
        out_h,
        out_w,
    };

    let selection = MatmulSelection {
        tile_shape,
        tile_count,
        plane_dim,
        rows_per_plane: 1,
    };
    let config_input = CompleteStageTiling {
        tile_shape: selection.tile_shape,
        tile_count: selection.tile_count,
    };

    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };
    test_convolution_algorithm::<A, Args, P, R>(
        client,
        problem,
        (config_input, STAGE_BUFFERING, vectorization), // TODO support double buffering
        selection,
    );
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! conv2d_standard_tests {
    () => {
        use $crate::convolution::tests::ConvolutionSize;
        use $crate::matmul::components::MatmulSize;

        mod t8x8x8 {
            use super::*;
            $crate::conv2d_standard_tests!(MatmulSize { m: 8, n: 8, k: 8 });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t16x16x16 {
            use super::*;
            $crate::conv2d_standard_tests!(MatmulSize {
                m: 16,
                n: 16,
                k: 16
            });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t32x8x16 {
            use super::*;
            $crate::conv2d_standard_tests!(MatmulSize { m: 32, n: 8, k: 16 });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t8x32x16 {
            use super::*;
            $crate::conv2d_standard_tests!(MatmulSize { m: 8, n: 32, k: 16 });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t16x16x8 {
            use super::*;
            $crate::conv2d_standard_tests!(MatmulSize { m: 16, n: 16, k: 8 });
        }
    };

    ($tile:expr) => {
        mod s1x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, MatmulSize { m: 1, n: 1, k: 1 });
        }

        mod s8x8x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, MatmulSize { m: 8, n: 8, k: 1 });
        }

        mod s2x2x2 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, MatmulSize { m: 2, n: 2, k: 2 });
        }

        mod s4x4x2 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, MatmulSize { m: 4, n: 4, k: 2 });
        }
    };

    ($tile:expr, $stage:expr) => {
        mod p4x4x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 4,
                    w: 4,
                    c: 1,
                    out_c: 1
                }
            );
        }

        mod p17x17x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 17,
                    w: 17,
                    c: 1,
                    out_c: 1
                }
            );
        }

        mod p16x16x16x32 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 16,
                    w: 16,
                    c: 16,
                    out_c: 32
                }
            );
        }

        mod p32x32x32x16 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 32,
                    w: 32,
                    c: 32,
                    out_c: 16
                }
            );
        }

        mod p64x32x32x128 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 64,
                    w: 32,
                    c: 32,
                    out_c: 128
                }
            );
        }

        mod p32x32x64x3 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 32,
                    w: 32,
                    c: 64,
                    out_c: 3
                }
            );
        }

        mod p100x100x100x100 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 100,
                    w: 100,
                    c: 100,
                    out_c: 100
                }
            );
        }

        mod p20x20x16x32 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 20,
                    w: 20,
                    c: 16,
                    out_c: 32
                }
            );
        }

        mod p23x10x17x20 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $stage,
                ConvolutionSize {
                    h: 23,
                    w: 10,
                    c: 17,
                    out_c: 20
                }
            );
        }
    };

    ($tile:expr, $stage:expr, $problem:expr) => {
        use $crate::convolution::algorithm::multi_stage_tma::MultiStageTmaConvAlgorithm;
        use $crate::convolution::algorithm::simple::SimpleConvAlgorithm;
        use $crate::convolution::algorithm::simple_tma::SimpleTmaConvAlgorithm;
        use $crate::matmul::components::global::args::{TensorArgs, TensorMapArgs};

        #[test]
        pub fn simple_coalesced_im2col() {
            cubecl_linalg::convolution::tests::test_algo::<
                SimpleConvAlgorithm<TMM>,
                TensorArgs,
                Precision,
                TestRuntime,
            >($tile, $stage, $problem);
        }

        #[test]
        pub fn simple_tma_im2col() {
            cubecl_linalg::convolution::tests::test_algo::<
                SimpleTmaConvAlgorithm<TMM>,
                TensorMapArgs,
                Precision,
                TestRuntime,
            >($tile, $stage, $problem);
        }

        #[test]
        pub fn multi_stage_tma_im2col() {
            cubecl_linalg::convolution::tests::test_algo::<
                MultiStageTmaConvAlgorithm<TMM>,
                TensorMapArgs,
                Precision,
                TestRuntime,
            >($tile, $stage, $problem);
        }
    };
}
