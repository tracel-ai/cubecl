#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_internal {
    ($i_16x16x16:ident, $i_32x8x16:ident, $i_8x32x16:ident, $eg:ty, $es:ty, $ea:ty, $plane_dim:expr) => {
        use cubecl_linalg::matmul::components::{
            batch,
            batch::one_to_one,
            global, global::homogeneous,
            stage, stage::row_accumulate,
            stage::{
                TilingOrderConfig, StageSize, S8x8x1, S8x1x1, S1x1x1, S1x1x2,
                S1x2x1, S2x1x1, S2x2x1, S2x2x2, S4x4x1, S4x4x2,
            },
            tile,
            tile::plane::{PlaneMma32x32x32, PlaneMma16x16x8, PlaneMma16x16x32},
            MatrixLayout,
            MatmulProblem,
            StageDim,
        };
        use std::marker::PhantomData;
        use cubecl_linalg::matmul::kernels::cmma_matmul::{MatmulLaunchDispatch, make_cmma_config, AdvancedConfig};
        use cubecl_linalg::matmul::tests::cmma_matmul::matmul_test_launcher::test_matmul_internal;
        use cubecl_core::prelude::*;

        type T = Config;
        type S = stage::row_accumulate::Config<T>;
        type G = global::homogeneous::Config<S>;
        type B = batch::one_to_one::Config<G>;

        macro_rules! matmul_test {
            (
                                                                        $test_name:ident,
                                                                        $problem:expr,
                                                                        $cube_dim:expr,
                                                                        $cube_count:expr,
                                                                        $stage_size:ty,
                                                                        $tile_matmul_type:ident,
                                                                        $advanced_config:expr
                                                                    ) => {
                pub fn $test_name<R: Runtime>(device: &R::Device) {
                    let problem = $problem;
                    struct D {}
                    impl MatmulLaunchDispatch for D {
                        const PLANE_DIM: u32 = $plane_dim;
                        type StageSize = $stage_size;
                        type ElementInput = $es;
                        type ElementAccumulator = $ea;
                        type TileConfig = Config;
                        type TileMatmul = $tile_matmul_type<ES, EA>;
                        fn cube_dim() -> CubeDim {
                            $cube_dim
                        }
                        fn cube_count<EG: Numeric>(_problem: &MatmulProblem<EG>) -> CubeCount {
                            $cube_count
                        }
                        fn tile_config<EG: Numeric>(
                            plane_dim: u32,
                            problem: &MatmulProblem<EG>,
                        ) -> Self::TileConfig {
                            Self::TileConfig::new(
                                plane_dim,
                                problem.lhs_layout,
                                problem.rhs_layout,
                                problem.lhs_line_size as u32,
                                problem.rhs_line_size as u32,
                                problem.out_line_size as u32,
                            )
                        }
                    }

                    type EG = $eg;
                    type ES = $es;
                    type EA = $ea;
                    type StageSize = $stage_size;

                    type TileMatmul = $tile_matmul_type<ES, EA>;
                    type StageMatmul = stage::row_accumulate::Matmul<ES, EG, EA, TileMatmul, StageSize, S>;
                    type GlobalMatmul = global::homogeneous::Matmul<EG, ES, StageMatmul, G>;
                    type BatchMatmul = batch::one_to_one::Matmul<EG, ES, GlobalMatmul, B>;

                    let config = make_cmma_config::<
                        EG,
                        D,
                    >(&problem, &$cube_dim, &$cube_count, &$advanced_config);

                    test_matmul_internal::<BatchMatmul, EG, B, G, R>(problem, $cube_dim, $cube_count, config, device);
                }
            };
        }

        #[test]
        pub fn test_batch_matmul_b3x4_g300x300x300_s4x4x2() {
            matmul_test!(
                test_batch_matmul_b3x4_g300x300x300_s4x4x2,
                MatmulProblem {
                    m: 300,
                    n: 300,
                    k: 300,
                    batches: vec![3, 4],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(5, 5, 12),
                S4x4x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3x4_g300x300x300_s4x4x2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3x4_g108x108x243_s4x4x2() {
            matmul_test!(
                test_batch_matmul_b3x4_g108x108x243_s4x4x2,
                MatmulProblem {
                    m: 108,
                    n: 108,
                    k: 243,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(2, 2, 1),
                S4x4x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3x4_g108x108x243_s4x4x2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3x4_g256x256x256_s4x4x2() {
            matmul_test!(
                test_batch_matmul_b3x4_g256x256x256_s4x4x2,
                MatmulProblem {
                    m: 256,
                    n: 256,
                    k: 256,
                    batches: vec![3, 4],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 2,
                    rhs_line_size: 2,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(4, 4, 12),
                S4x4x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3x4_g256x256x256_s4x4x2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3_g256x256x256_s4x4x2() {
            matmul_test!(
                test_batch_matmul_b3_g256x256x256_s4x4x2,
                MatmulProblem {
                    m: 256,
                    n: 256,
                    k: 256,
                    batches: vec![3],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(4, 4, 3),
                S4x4x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3_g256x256x256_s4x4x2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3_g16x16x16_s1x1x1_col_major() {
            matmul_test!(
                test_batch_matmul_b3_g16x16x16_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![3],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 3),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3_g16x16x16_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3_g16x16x16_s1x1x1() {
            matmul_test!(
                test_batch_matmul_b3_g16x16x16_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![3],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 3),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3_g16x16x16_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_g256x256x256_s4x4x2() {
            matmul_test!(
                test_batch_matmul_g256x256x256_s4x4x2,
                MatmulProblem {
                    m: 256,
                    n: 256,
                    k: 256,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(4, 4, 1),
                S4x4x2,
                $i_16x16x16,
                AdvancedConfig {
                    tiling_order: TilingOrderConfig::YMajor
                }
            );
            test_batch_matmul_g256x256x256_s4x4x2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_g32x32x32_s1x1x1_col_y_major() {
            matmul_test!(
                test_batch_matmul_g32x32x32_s1x1x1,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(2, 2, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig {
                    tiling_order: TilingOrderConfig::YMajor
                }
            );
            test_batch_matmul_g32x32x32_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_g32x32x32_s1x1x1() {
            matmul_test!(
                test_batch_matmul_g32x32x32_s1x1x1,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(2, 2, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g32x32x32_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_g16x14x16_s1x1x1_rhs_col_major() {
            matmul_test!(
                test_batch_matmul_g14x16x16_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 14,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 2,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g14x16x16_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_g16x12x16_s1x1x1() {
            matmul_test!(
                test_batch_matmul_g14x16x16_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 12,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g14x16x16_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_g16x16x12_s1x1x1() {
            matmul_test!(
                test_batch_matmul_g14x16x16_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 12,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g14x16x16_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g60x60x120_s4x4x2() {
            matmul_test!(
                test_global_matmul_g60x60x120_s4x4x2,
                MatmulProblem {
                    m: 60,
                    n: 60,
                    k: 120,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(1, 1, 1),
                S4x4x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_global_matmul_g60x60x120_s4x4x2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x36_s1x1x1() {
            matmul_test!(
                test_global_matmul_g16x16x36_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 36,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x36_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g12x12x16_s1x1x1() {
            matmul_test!(
                test_global_matmul_g12x12x16_s1x1x1,
                MatmulProblem {
                    m: 12,
                    n: 12,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_global_matmul_g12x12x16_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s1x1x1_unlined() {
            matmul_test!(
                test_global_matmul_g16x16x16_s1x1x1_unlined,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 1,
                    rhs_line_size: 1,
                    out_line_size: 1,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s1x1x1_unlined::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_tile_t16x16x16_row_col() {
            matmul_test!(
                test_tile_16_16_16,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_tile_16_16_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_tile_t16x16x16_col_row() {
            matmul_test!(
                test_tile_t16x16x16_col_row,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_tile_t16x16x16_col_row::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_tile_t16x16x16_col_col() {
            matmul_test!(
                test_tile_t16x16x16_col_col,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_tile_t16x16x16_col_col::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_tile_t32x8x16() {
            matmul_test!(
                test_tile_t32x8x16,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_32x8x16,
                AdvancedConfig::default()
            );

            test_tile_t32x8x16::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_tile_t32x8x16_row_col() {
            matmul_test!(
                test_tile_t32x8x16_row_col,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 1,
                    rhs_line_size: 1,
                    out_line_size: 1,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_32x8x16,
                AdvancedConfig::default()
            );

            test_tile_t32x8x16_row_col::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_tile_t32x8x16_col_row() {
            matmul_test!(
                test_tile_t32x8x16_col_row,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_32x8x16,
                AdvancedConfig::default()
            );

            test_tile_t32x8x16_col_row::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_tile_t32x8x16_col_col() {
            matmul_test!(
                test_tile_t32x8x16_col_col,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_32x8x16,
                AdvancedConfig::default()
            );

            test_tile_t32x8x16_col_col::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_tile_t8x32x16() {
            matmul_test!(
                test_tile_t8x32x16,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_8x32x16,
                AdvancedConfig::default()
            );

            test_tile_t8x32x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_tile_t8x32x16_row_col() {
            matmul_test!(
                test_tile_t8x32x16_row_col,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_8x32x16,
                AdvancedConfig::default()
            );

            test_tile_t8x32x16_row_col::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_tile_t8x32x16_col_row() {
            matmul_test!(
                test_tile_t8x32x16_col_row,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_8x32x16,
                AdvancedConfig::default()
            );

            test_tile_t8x32x16_col_row::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_tile_t8x32x16_col_col() {
            matmul_test!(
                test_tile_t8x32x16_col_col,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new(32, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_8x32x16,
                AdvancedConfig::default()
            );

            test_tile_t8x32x16_col_col::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s1x1x1_line2() {
            matmul_test!(
                test_global_matmul_g16x16x16_s1x1x1_line2,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 2,
                    rhs_line_size: 2,
                    out_line_size: 2,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s1x1x1_line2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s1x1x1() {
            matmul_test!(
                test_global_matmul_g16x16x16_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_ymajor() {
            matmul_test!(
                test_global_matmul_ymajor,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x2,
                $i_16x16x16,
                AdvancedConfig {
                    tiling_order: TilingOrderConfig::YMajor
                }
            );

            test_global_matmul_ymajor::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x32_s1x1x1() {
            matmul_test!(
                test_global_matmul_g16x16x32_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x32_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s1x1x1_col_major() {
            matmul_test!(
                test_global_matmul_g16x16x16_s1x1x1_col_major,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s1x1x1_col_major::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x128_s1x1x1() {
            matmul_test!(
                test_global_matmul_g16x16x128_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 128,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x128_s1x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g32x16x128_s2x1x1() {
            matmul_test!(
                test_global_matmul_g32x16x128_s2x1x1,
                MatmulProblem {
                    m: 32,
                    n: 16,
                    k: 128,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g32x16x128_s2x1x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g32x32x224_s2x2x2() {
            matmul_test!(
                test_global_matmul_g32x32x224_s2x2x2,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 224,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g32x32x224_s2x2x2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x32x16_s1x2x1() {
            matmul_test!(
                test_global_matmul_g16x32x16_s1x2x1,
                MatmulProblem {
                    m: 16,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x2x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x32x16_s1x2x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_col_major_tiling() {
            matmul_test!(
                test_global_matmul_col_major_tiling,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_col_major_tiling::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g32x32x16_s2x2x1() {
            matmul_test!(
                test_global_matmul_g32x32x16_s2x2x1,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g32x32x16_s2x2x1::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_s1x2x1() {
            matmul_test!(
                test_stage_matmul_s1x2x1,
                MatmulProblem {
                    m: 16,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x2x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s1x2x1::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s1x1x1() {
            matmul_test!(
                test_stage_matmul_s1x1x1,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s1x1x1::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s2x2x2_row_col() {
            matmul_test!(
                test_stage_matmul_s2x2x2_row_col,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s2x2x2_row_col::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s2x2x2_col_row() {
            matmul_test!(
                test_stage_matmul_s2x2x2_col_row,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s2x2x2_col_row::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s2x2x2_col_col() {
            matmul_test!(
                test_stage_matmul_s2x2x2_col_col,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s2x2x2_col_col::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s2x1x1() {
            matmul_test!(
                test_stage_matmul_s2x1x1,
                MatmulProblem {
                    m: 32,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s2x1x1::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s1x1x1_t32x8x16_col_major() {
            matmul_test!(
                test_stage_matmul_s1x1x1_col_major,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 1,
                    rhs_line_size: 1,
                    out_line_size: 1,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_32x8x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s1x1x1_col_major::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s8x1x1() {
            matmul_test!(
                test_stage_matmul_s8x1x1,
                MatmulProblem {
                    m: 128,
                    n: 16,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 1,
                    rhs_line_size: 1,
                    out_line_size: 1,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 8, 1),
                CubeCount::Static(1, 1, 1),
                S8x1x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s8x1x1::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s4x4x1() {
            matmul_test!(
                test_stage_matmul_s4x4x1,
                MatmulProblem {
                    m: 64,
                    n: 64,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(1, 1, 1),
                S4x4x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s4x4x1::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s4x4x2() {
            matmul_test!(
                test_stage_matmul_s4x4x2,
                MatmulProblem {
                    m: 64,
                    n: 64,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 4, 1),
                CubeCount::Static(1, 1, 1),
                S4x4x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s4x4x2::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s2x2x1() {
            matmul_test!(
                test_stage_matmul_s2x2x1,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x1,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s2x2x1::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s2x2x2() {
            matmul_test!(
                test_stage_matmul_s2x2x2,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 2, 1),
                CubeCount::Static(1, 1, 1),
                S2x2x2,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s2x2x2::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s1x1x1_t32x8x16_row_major() {
            matmul_test!(
                test_stage_matmul_s1x1x1,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_32x8x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s1x1x1::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s1x1x1_t8x32x16() {
            matmul_test!(
                test_stage_matmul_s1x1x1,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    batches: vec![],
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                    _element: PhantomData,
                },
                CubeDim::new($plane_dim, 1, 1),
                CubeCount::Static(1, 1, 1),
                S1x1x1,
                $i_8x32x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s1x1x1::<TestRuntime>(&Default::default());
        }
    };
}
