use crate::base::ConvolutionProblem;
use crate::tests::convolution_test_launcher::test_convolution_algorithm;
use crate::{
    algorithm::Algorithm, args::ConvInputsLaunch, base::Dimensionality,
    tests::test_utils::TestPrecision,
};
use cubecl_core::Runtime;
use cubecl_matmul::components::global::args::ConcreteOutputFactory;
use cubecl_matmul::components::global::args::MatmulArgs;
use cubecl_matmul::components::stage::PartitionBuffering;
use cubecl_matmul::components::{MatrixLayout, PartitionSize, StageSize, TileSize, TilingScheme};
use cubecl_matmul::kernels::layered::MatmulSelection;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionSize {
    pub h: usize,
    pub w: usize,
    pub c: usize,

    pub out_c: usize,
}

pub fn test_algo<A: Algorithm, Args: MatmulArgs, P: TestPrecision, R: Runtime>(
    tile_size: TileSize,
    partition_size: PartitionSize,
    stage_size: StageSize,
    problem: ConvolutionSize,
) where
    Args::Input<P::EG>: ConvInputsLaunch,
    Args::Output<P::EG>: ConcreteOutputFactory,
{
    let client = R::client(&Default::default());
    let plane_dim = client.properties().hardware.plane_size_max;

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

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(partition_size)
        .with_stage_size(stage_size)
        .build()
        .unwrap();

    let selection = MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .build();

    test_convolution_algorithm::<A, Args, P, R>(client, problem, selection);
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
        use cubecl_matmul::components::{PartitionSize, StageSize, TileSize};
        use $crate::tests::ConvolutionSize;

        mod t8x8x8 {
            use super::*;
            $crate::conv2d_standard_tests!(TileSize { m: 8, n: 8, k: 8 });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t16x16x16 {
            use super::*;
            $crate::conv2d_standard_tests!(TileSize {
                m: 16,
                n: 16,
                k: 16
            });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t32x8x16 {
            use super::*;
            $crate::conv2d_standard_tests!(TileSize { m: 32, n: 8, k: 16 });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t8x32x16 {
            use super::*;
            $crate::conv2d_standard_tests!(TileSize { m: 8, n: 32, k: 16 });
        }

        #[cfg(not(all(feature = "msl", target_os = "macos")))]
        mod t16x16x8 {
            use super::*;
            $crate::conv2d_standard_tests!(TileSize { m: 16, n: 16, k: 8 });
        }
    };

    ($tile:expr) => {
        mod p1x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, PartitionSize { m: 1, n: 1, k: 1 });
        }

        mod p1x8x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, PartitionSize { m: 1, n: 8, k: 1 });
        }

        mod p1x2x2 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, PartitionSize { m: 1, n: 2, k: 2 });
        }

        mod p1x4x2 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, PartitionSize { m: 1, n: 4, k: 2 });
        }
    };

    ($tile:expr, $partition:expr) => {
        mod s1x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, $partition, StageSize { m: 1, n: 1, k: 1 });
        }

        mod s2x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, $partition, StageSize { m: 2, n: 1, k: 1 });
        }

        mod s4x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, $partition, StageSize { m: 4, n: 1, k: 1 });
        }

        mod s8x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!($tile, $partition, StageSize { m: 8, n: 1, k: 1 });
        }
    };

    ($tile:expr, $partition:expr, $stage:expr) => {
        mod g4x4x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 4,
                    w: 4,
                    c: 1,
                    out_c: 1
                }
            );
        }

        mod g17x17x1x1 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 17,
                    w: 17,
                    c: 1,
                    out_c: 1
                }
            );
        }

        mod g16x16x16x32 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 16,
                    w: 16,
                    c: 16,
                    out_c: 32
                }
            );
        }

        mod g32x32x32x16 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 32,
                    w: 32,
                    c: 32,
                    out_c: 16
                }
            );
        }

        mod g64x32x32x128 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 64,
                    w: 32,
                    c: 32,
                    out_c: 128
                }
            );
        }

        mod g32x32x64x3 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 32,
                    w: 32,
                    c: 64,
                    out_c: 3
                }
            );
        }

        mod g100x100x100x100 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 100,
                    w: 100,
                    c: 100,
                    out_c: 100
                }
            );
        }

        mod g20x20x16x32 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
                $stage,
                ConvolutionSize {
                    h: 20,
                    w: 20,
                    c: 16,
                    out_c: 32
                }
            );
        }

        mod g23x10x17x20 {
            use super::*;
            $crate::conv2d_standard_tests!(
                $tile,
                $partition,
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

    ($tile:expr, $partition:expr, $stage:expr, $problem:expr) => {
        use cubecl_matmul::components::global::args::{TensorArgs, TensorMapArgs};
        use $crate::algorithm::multi_stage_tma::MultiStageTmaConvAlgorithm;
        use $crate::algorithm::simple::SimpleConvAlgorithm;
        use $crate::algorithm::simple_tma::SimpleTmaConvAlgorithm;

        #[test]
        pub fn simple_coalesced_im2col() {
            cubecl_convolution::tests::test_algo::<
                SimpleConvAlgorithm<TMM>,
                TensorArgs,
                Precision,
                TestRuntime,
            >($tile, $partition, $stage, $problem);
        }

        #[test]
        pub fn simple_tma_im2col() {
            cubecl_convolution::tests::test_algo::<
                SimpleTmaConvAlgorithm<TMM>,
                TensorMapArgs,
                Precision,
                TestRuntime,
            >($tile, $partition, $stage, $problem);
        }

        #[test]
        pub fn multi_stage_tma_im2col() {
            cubecl_convolution::tests::test_algo::<
                MultiStageTmaConvAlgorithm<TMM>,
                TensorMapArgs,
                Precision,
                TestRuntime,
            >($tile, $partition, $stage, $problem);
        }
    };
}
