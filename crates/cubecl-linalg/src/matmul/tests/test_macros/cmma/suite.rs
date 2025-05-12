use crate::matmul::components::stage::StageVectorization;
use crate::matmul::components::{CompleteStageTiling, MatmulProblem, MatrixLayout};
use crate::matmul::components::{MatmulSelection, MatmulSize};
use crate::matmul::kernels::matmul::Algorithm;
use crate::matmul::tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm;
use crate::matmul::tests::cmma_matmul::tma_test_launcher::test_tma_matmul_algorithm;
use crate::matmul::tests::test_utils::TestPrecision;
use cubecl_core::Runtime;

pub fn test_algo<A: Algorithm, P: TestPrecision, R: Runtime>(
    layouts: (MatrixLayout, MatrixLayout),
    tile_shape: MatmulSize,
    tile_count: MatmulSize,
    problem: MatmulSize,
    rows_per_plane: u32,
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
        tile_shape,
        tile_count,
        plane_dim,
        rows_per_plane,
    };
    let config_input = CompleteStageTiling {
        tile_shape: selection.tile_shape,
        tile_count: selection.tile_count,
    };
    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };

    test_matmul_algorithm::<A, P, R>(
        client,
        problem,
        (
            (
                config_input,
                A::stage_buffering_strategy(),
                vectorization,
                A::num_stages(),
            ),
            A::loading_precompute_strategy(),
        ),
        selection,
    );
}

pub fn test_algo_tma<A: Algorithm, P: TestPrecision, R: Runtime>(
    layouts: (MatrixLayout, MatrixLayout),
    tile_shape: MatmulSize,
    tile_count: MatmulSize,
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
    test_tma_matmul_algorithm::<A, P, R>(
        client,
        problem,
        (
            (
                config_input,
                A::stage_buffering_strategy(),
                vectorization,
                A::num_stages(),
            ),
            A::loading_precompute_strategy(),
        ),
        selection,
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! matmul_standard_tests {
    (standard; $lhs_layout:ident, $rhs_layout:ident, $tile:expr, $stage:expr, $problem:expr) => {
        use $crate::matmul::components::global::load::{
            async_full_cyclic, async_full_maximize_slice_length, async_full_maximize_unit_count, sync_full_strided, sync_full_tilewise, async_full_cooperative,
        };
        use $crate::matmul::components::stage::{ColMajorTilingOrder, RowMajorTilingOrder};
        use $crate::matmul::kernels::matmul::double_buffering::{CyclicDoubleBufferingAlgorithm, TilewiseDoubleBufferingAlgorithm, HybridDoubleBufferingAlgorithm};
        use $crate::matmul::kernels::matmul::simple::SimpleAlgorithm;
        use $crate::matmul::kernels::matmul::simple_unit::SimpleUnitAlgorithm;
        use $crate::matmul::kernels::matmul::simple_barrier::SimpleBarrierAlgorithm;

        #[test]
        pub fn simple_cyclic() {
            cubecl_linalg::matmul::tests::test_algo::<SimpleAlgorithm<TMM>, Precision, TestRuntime>(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn simple_cyclic_multi_rows() {
            cubecl_linalg::matmul::tests::test_algo::<SimpleAlgorithm<TMM>, Precision, TestRuntime>(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                2,
            );
        }

        #[test]
        pub fn simple_strided() {
            cubecl_linalg::matmul::tests::test_algo::<
                SimpleAlgorithm<TMM, sync_full_strided::LoadingStrategy, sync_full_strided::LoadingStrategy>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn simple_tilewise() {
            cubecl_linalg::matmul::tests::test_algo::<
                SimpleAlgorithm<TMM, sync_full_tilewise::LoadingStrategy<RowMajorTilingOrder>, sync_full_tilewise::LoadingStrategy<ColMajorTilingOrder>>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn simple_barrier_cooperative() {
            cubecl_linalg::matmul::tests::test_algo::<
                SimpleBarrierAlgorithm<TMM, async_full_cooperative::LoadingStrategy>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn simple_barrier_cyclic() {
            cubecl_linalg::matmul::tests::test_algo::<
                SimpleBarrierAlgorithm<TMM, async_full_cyclic::LoadingStrategy<ColMajorTilingOrder>>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn simple_barrier_maximize_slice_length() {
            cubecl_linalg::matmul::tests::test_algo::<
                SimpleBarrierAlgorithm<TMM, async_full_maximize_slice_length::LoadingStrategy>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn simple_barrier_maximize_unit_count() {
            cubecl_linalg::matmul::tests::test_algo::<
                SimpleBarrierAlgorithm<TMM, async_full_maximize_unit_count::LoadingStrategy>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn double_buffering_single_row_cyclic() {
            cubecl_linalg::matmul::tests::test_algo::<
                CyclicDoubleBufferingAlgorithm<TMM>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn double_buffering_multi_row_cyclic() {
            cubecl_linalg::matmul::tests::test_algo::<
                CyclicDoubleBufferingAlgorithm<TMM>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                2,
            );
        }

        #[test]
        pub fn double_buffering_single_row_tilewise() {
            cubecl_linalg::matmul::tests::test_algo::<
                TilewiseDoubleBufferingAlgorithm<TMM>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn double_buffering_multi_row_tilewise() {
            cubecl_linalg::matmul::tests::test_algo::<
                TilewiseDoubleBufferingAlgorithm<TMM>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                2,
            );
        }

        #[test]
        pub fn double_buffering_single_row_hybrid() {
            cubecl_linalg::matmul::tests::test_algo::<
                HybridDoubleBufferingAlgorithm<TMM>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }

        #[test]
        pub fn double_buffering_multi_row_hybrid() {
            cubecl_linalg::matmul::tests::test_algo::<
                HybridDoubleBufferingAlgorithm<TMM>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                2,
            );
        }

        #[test]
        pub fn simple_unit() {
            cubecl_linalg::matmul::tests::test_algo::<
                SimpleUnitAlgorithm,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
                1,
            );
        }
    };

    (tma; $lhs_layout:ident, $rhs_layout:ident, $tile:expr, $stage:expr, $problem:expr) => {
        use $crate::matmul::kernels::matmul::simple_tma::SimpleTmaAlgorithm;

        #[test]
        pub fn simple_tma() {
            cubecl_linalg::matmul::tests::test_algo_tma::<
                SimpleTmaAlgorithm<TMM>,
                Precision,
                TestRuntime,
            >(
                (MatrixLayout::$lhs_layout, MatrixLayout::$rhs_layout),
                $tile,
                $stage,
                $problem,
            );
        }
    };

    ($kind: ident) => {
        use $crate::matmul::components::{MatmulSize, MatrixLayout};

        mod row_major {
            use super::*;

            mod row_major {
                use super::*;
                $crate::matmul_standard_tests!($kind; RowMajor, RowMajor);
            }

            mod col_major {
                use super::*;
                $crate::matmul_standard_tests!($kind; RowMajor, ColMajor);
            }
        }

        mod col_major {
            use super::*;

            mod row_major {
                use super::*;
                $crate::matmul_standard_tests!($kind; ColMajor, RowMajor);
            }

            mod col_major {
                use super::*;
                $crate::matmul_standard_tests!($kind; ColMajor, ColMajor);
            }
        }
    };

    ($kind: ident; $lhs_layout:ident, $rhs_layout:ident) => {
        mod t8x8x8 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                MatmulSize { m: 8, n: 8, k: 8 }
            );
        }

        mod t16x16x16 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                MatmulSize {
                    m: 16,
                    n: 16,
                    k: 16
                }
            );
        }

        mod t32x8x16 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                MatmulSize { m: 32, n: 8, k: 16 }
            );
        }

        mod t8x32x16 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                MatmulSize { m: 8, n: 32, k: 16 }
            );
        }

        mod t16x16x8 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                MatmulSize { m: 16, n: 16, k: 8 }
            );
        }
    };

    ($kind: ident; $lhs_layout:ident, $rhs_layout:ident, $tile:expr) => {
        mod s1x1x1 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                MatmulSize { m: 1, n: 1, k: 1 }
            );
        }

        mod s8x8x1 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                MatmulSize { m: 8, n: 8, k: 1 }
            );
        }

        mod s16x16x1 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                MatmulSize { m: 16, n: 16, k: 1 }
            );
        }

        mod s2x2x2 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                MatmulSize { m: 2, n: 2, k: 2 }
            );
        }

        mod s8x8x4 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                MatmulSize { m: 8, n: 8, k: 4 }
            );
        }

        mod s4x4x2 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                MatmulSize { m: 4, n: 4, k: 2 }
            );
        }
    };

    ($kind: ident; $lhs_layout:ident, $rhs_layout:ident, $tile:expr, $stage:expr) => {
        mod p8x8x8 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 8,
                    n: 8,
                    k: 8
                }
            );
        }

        mod p16x16x16 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 16,
                    n: 16,
                    k: 16
                }
            );
        }

        mod p32x32x32 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 32,
                    n: 32,
                    k: 32
                }
            );
        }

        mod p64x32x32 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 64,
                    n: 32,
                    k: 32
                }
            );
        }

        mod p32x32x64 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 32,
                    n: 32,
                    k: 64
                }
            );
        }

        mod p100x100x100 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 100,
                    n: 100,
                    k: 100
                }
            );
        }

        mod p20x20x16 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 20,
                    n: 20,
                    k: 16
                }
            );
        }

        mod p23x1x17 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize { m: 23, n: 1, k: 17 }
            );
        }

        mod p256x256x256 {
            use super::*;
            $crate::matmul_standard_tests!(
                $kind;
                $lhs_layout,
                $rhs_layout,
                $tile,
                $stage,
                MatmulSize {
                    m: 256,
                    n: 256,
                    k: 256
                }
            );
        }
    };
}
