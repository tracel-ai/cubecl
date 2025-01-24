use std::fmt::Display;

use crate::matmul::components::stage::CommonStageInput;
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{MatmulProblem, MatrixLayout};
use crate::matmul::components::{MatmulSelection, MatmulSize};
use crate::matmul::kernels::matmul::Algorithm;
use crate::matmul::tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm;
use crate::matmul::tests::test_utils::CastInto;

use cubecl_core::prelude::Float;
use cubecl_core::{CubeElement, Runtime};

pub fn test_algo<A: Algorithm<Selection = MatmulSelection>, P: TestPrecision, R: Runtime>(
    layouts: (MatrixLayout, MatrixLayout),
    tile: MatmulSize,
    stage: MatmulSize,
    problem: MatmulSize,
) {
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

    let problem = MatmulProblem {
        m: problem.m as usize,
        n: problem.n as usize,
        k: problem.k as usize,
        batches: (vec![2], vec![2]),
        lhs_layout: layouts.0,
        rhs_layout: layouts.1,
        lhs_line_size: 1, // Will be changed
        rhs_line_size: 1, // Will be changed
        out_line_size: 1, // Will be changed
    };

    let selection = MatmulSelection {
        tile,
        num_stages: stage,
        plane_dim,
    };
    let config_input = CommonStageInput {
        tile: A::TileMatmul::input(selection.tile),
        num_stages: selection.num_stages,
    };

    test_matmul_algorithm::<A, P::EG, P::ES, R>(client, problem, config_input, selection);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! matmul_standard_tests {
    () => {
        use $crate::matmul::components::{MatmulSize, MatrixLayout};

        mod row_major {
            use super::*;

            mod row_major {
                use super::*;
                $crate::matmul_standard_tests!(RowMajor, RowMajor);
            }

            mod col_major {
                use super::*;
                $crate::matmul_standard_tests!(RowMajor, ColMajor);
            }
        }

        mod col_major {
            use super::*;

            mod row_major {
                use super::*;
                $crate::matmul_standard_tests!(ColMajor, RowMajor);
            }

            mod col_major {
                use super::*;
                $crate::matmul_standard_tests!(ColMajor, ColMajor);
            }
        }
    };

    ($lhs_layout:ident, $rhs_layout:ident) => {
        mod t16x16x16 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, MatmulSize { m: 16, n: 16, k: 16 });
        }

        mod t32x8x16 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, MatmulSize { m: 32, n: 8, k: 16 });
        }

        mod t8x32x16 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, MatmulSize { m: 8, n: 32, k: 16 });
        }

        mod t16x16x8 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, MatmulSize { m: 16, n: 16, k: 8 });
        }
    };

    ($lhs_layout:ident, $rhs_layout:ident, $tile:expr) => {
        mod s1x1x1 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, MatmulSize { m: 1, n: 1, k: 1 });
        }

        mod s8x8x1 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, MatmulSize { m: 8, n: 8, k: 1 });
        }

        mod s2x2x2 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, MatmulSize { m: 2, n: 2, k: 2 });
        }

        mod s4x4x2 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, MatmulSize { m: 4, n: 4, k: 2 });
        }
    };

    ($lhs_layout:ident, $rhs_layout:ident, $tile:expr, $stage:expr) => {
        mod p32x32x32 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, $stage, MatmulSize { m: 32, n: 32, k: 32 });
        }

        mod p64x32x32 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, $stage, MatmulSize { m: 64, n: 32, k: 32 });
        }

        mod p32x32x64 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, $stage, MatmulSize { m: 32, n: 32, k: 64 });
        }

        mod p100x100x100 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, $stage, MatmulSize { m: 100, n: 100, k: 100 });
        }

        mod p20x20x16 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, $stage, MatmulSize { m: 65, n: 16, k: 16 });
        }

        mod p23x1x17 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, $stage, MatmulSize { m: 23, n: 1, k: 17 });
        }

        mod p256x256x256 {
            use super::*;
            $crate::matmul_standard_tests!($lhs_layout, $rhs_layout, $tile, $stage, MatmulSize { m: 256, n: 256, k: 256 });
        }
    };

    ($lhs_layout:ident, $rhs_layout:ident, $tile:expr, $stage:expr, $problem:expr) => {
        use $crate::matmul::kernels::matmul::standard::StandardAlgorithm;
        use $crate::matmul::kernels::matmul::specialized::SpecializedAlgorithm;
        use $crate::matmul::kernels::matmul::pipelined::PipelinedAlgorithm;

        #[test]
        pub fn standard() {
            cubecl_linalg::matmul::tests::test_algo::<StandardAlgorithm<TMM>, (EG, ES), TestRuntime>(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
            );
        }

        // #[test]
        // pub fn specialized() {
        //     cubecl_linalg::matmul::tests::test_algo::<SpecializedAlgorithm<TMM>, (EG, ES), TestRuntime>(
        //         (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
        //         $tile,
        //         $stage,
        //         $problem,
        //     );
        // }

        // #[test]
        // pub fn pipelined() {
        //     cubecl_linalg::matmul::tests::test_algo::<PipelinedAlgorithm<TMM>, (EG, ES), TestRuntime>(
        //         (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
        //         $tile,
        //         $stage,
        //         $problem,
        //     );
        // }
    };
}

pub trait TestPrecision {
    type EG: Float + CubeElement + Display + CastInto<Self::ES>;
    type ES: Float + CubeElement + Display + CastInto<Self::EG>;
}

impl<EG, ES> TestPrecision for (EG, ES)
where
    EG: Float + CubeElement + Display + CastInto<ES>,
    ES: Float + CubeElement + Display + CastInto<EG>,
{
    type EG = EG;
    type ES = ES;
}
