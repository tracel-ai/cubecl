#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_internal {
    ($i_16x16x16:ident, $i_32x8x16:ident, $i_8x32x16:ident, $eg:ty, $es:ty, $ea:ty, $plane_dim:expr) => {
        use cubecl_linalg::matmul::cmma_matmul::batch::CmmaBatchMatmul;
        use cubecl_linalg::matmul::cmma_matmul::batch::CmmaBatchMatmulConfig;
        use cubecl_linalg::matmul::cmma_matmul::global::CmmaGlobalMatmul;
        use cubecl_linalg::matmul::cmma_matmul::global::CmmaGlobalMatmulConfig;
        use cubecl_linalg::matmul::cmma_matmul::global::{
            LhsTensorLoader, RhsTensorLoader, TensorUnloader,
        };
        use cubecl_linalg::matmul::cmma_matmul::stage::CmmaStageMatmulConfig;
        use cubecl_linalg::matmul::cmma_matmul::stage::TilingOrderConfig;
        use cubecl_linalg::matmul::cmma_matmul::stage::{
            CmmaStageMatmul, S8x8x1, S8x1x1, S1x1x1, S1x1x2, S1x2x1, S2x1x1,
            S2x2x1, S2x2x2, S4x4x1, S4x4x2,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::{
            $i_16x16x16, $i_32x8x16, $i_8x32x16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::{PlaneMma32x32x32, PlaneMma16x16x8, PlaneMma16x16x32};
        use cubecl_linalg::matmul::cmma_matmul::tile::CmmaTileMatmulConfig;
        use cubecl_linalg::matmul::matmul_stage::StageMatmul;
        use cubecl_linalg::matmul::matmul_tile::TileMatmul;
        use cubecl_linalg::matmul::matrix::MatrixLayout;
        use cubecl_linalg::matmul::problem::MatmulProblem;
        use cubecl_linalg::matmul::stage_dim::StageDim;
        use cubecl_linalg::matmul::cmma_matmul::launch::create_stage_dim;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_matmul_internal;
        use cubecl_linalg::matmul::cmma_matmul::launch::make_cmma_config;
        use cubecl_linalg::matmul::cmma_matmul::launch::AdvancedConfig;
        use std::marker::PhantomData;

        type T = CmmaTileMatmulConfig;
        type S = CmmaStageMatmulConfig<T>;
        type G = CmmaGlobalMatmulConfig<S>;
        type B = CmmaBatchMatmulConfig<G>;

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
                    type T = CmmaTileMatmulConfig;
                    type S = CmmaStageMatmulConfig<T>;
                    type G = CmmaGlobalMatmulConfig<S>;

                    let problem = $problem;

                    type EG = $eg;
                    type ES = $es;
                    type EA = $ea;
                    type StageSize = $stage_size;

                    type TileMatmul = $tile_matmul_type<ES, EA, T>;
                    type StageMatmul = CmmaStageMatmul<ES, EG, EA, TileMatmul, StageSize, S>;
                    type GlobalMatmul = CmmaGlobalMatmul<EG, ES, StageMatmul, G>;
                    type BatchMatmul = CmmaBatchMatmul<EG, ES, GlobalMatmul, B>;

                    let config = make_cmma_config::<
                        EG,
                        ES,
                        EA,
                        TileMatmul,
                        StageMatmul,
                        GlobalMatmul,
                        BatchMatmul,
                        R,
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
                    b: vec![3, 4],
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
                    b: vec![],
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
                    b: vec![3, 4],
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
                    b: vec![3],
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
                    b: vec![3],
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
                    b: vec![3],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
        pub fn test_plane_mma_32x32x32() {
            matmul_test!(
                test_plane_mma_32x32x32,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    b: vec![],
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
                PlaneMma32x32x32,
                AdvancedConfig::default()
            );

            test_plane_mma_32x32x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t16x16x16_row_col() {
            matmul_test!(
                test_plane_mma_16_16_16,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_16_16_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t16x16x16_col_row() {
            matmul_test!(
                test_plane_mma_t16x16x16_col_row,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t16x16x16_col_row::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t16x16x16_col_col() {
            matmul_test!(
                test_plane_mma_t16x16x16_col_col,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t16x16x16_col_col::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t16x16x8() {
            matmul_test!(
                test_plane_mma_t16x16x8,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 8,
                    b: vec![],
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
                PlaneMma16x16x8,
                AdvancedConfig::default()
            );

            test_plane_mma_t16x16x8::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t16x16x32() {
            matmul_test!(
                test_plane_mma_t16x16x32,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 32,
                    b: vec![],
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
                PlaneMma16x16x32,
                AdvancedConfig::default()
            );

            test_plane_mma_t16x16x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t32x8x16() {
            matmul_test!(
                test_plane_mma_t32x8x16,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t32x8x16::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_plane_mma_t32x8x16_row_col() {
            matmul_test!(
                test_plane_mma_t32x8x16_row_col,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t32x8x16_row_col::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_plane_mma_t32x8x16_col_row() {
            matmul_test!(
                test_plane_mma_t32x8x16_col_row,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t32x8x16_col_row::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_plane_mma_t32x8x16_col_col() {
            matmul_test!(
                test_plane_mma_t32x8x16_col_col,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t32x8x16_col_col::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t8x32x16() {
            matmul_test!(
                test_plane_mma_t8x32x16,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t8x32x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_plane_mma_t8x32x16_row_col() {
            matmul_test!(
                test_plane_mma_t8x32x16_row_col,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t8x32x16_row_col::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_plane_mma_t8x32x16_col_row() {
            matmul_test!(
                test_plane_mma_t8x32x16_col_row,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t8x32x16_col_row::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn test_plane_mma_t8x32x16_col_col() {
            matmul_test!(
                test_plane_mma_t8x32x16_col_col,
                MatmulProblem {
                    m: 8,
                    n: 32,
                    k: 16,
                    b: vec![],
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

            test_plane_mma_t8x32x16_col_col::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s1x1x1_line2() {
            matmul_test!(
                test_global_matmul_g16x16x16_s1x1x1_line2,
                MatmulProblem {
                    m: 16,
                    n: 16,
                    k: 16,
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
        }

        #[test]
        pub fn test_stage_matmul_s2x2x2_row_col() {
            matmul_test!(
                test_stage_matmul_s2x2x2_row_col,
                MatmulProblem {
                    m: 32,
                    n: 32,
                    k: 32,
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
                    b: vec![],
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
