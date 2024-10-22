#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal {
    ($i_16x16x16:ident, $i_32x8x16:ident, $i_8x32x16:ident, $eg:ty, $es:ty, $ea:ty) => {
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
            CmmaStageMatmul, S128x128x16, S128x16x16, S16x16x16, S16x16x32, S16x32x16, S32x16x16,
            S32x32x16, S32x32x32, S32x8x16, S64x64x16, S64x64x32, S8x32x16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::{
            $i_16x16x16, $i_32x8x16, $i_8x32x16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::CmmaTileMatmulConfig;
        use cubecl_linalg::matmul::matmul_stage::StageMatmul;
        use cubecl_linalg::matmul::matmul_tile::TileMatmul;
        use cubecl_linalg::matmul::matrix::MatrixLayout;
        use cubecl_linalg::matmul::problem::MatmulProblem;
        use cubecl_linalg::matmul::stage_dim::StageDim;
        use cubecl_linalg::matmul::tests::create_stage_dim;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_matmul;
        use cubecl_linalg::matmul::tests::run_matmul_test;
        use cubecl_linalg::matmul::tests::AdvancedConfig;

        type T = CmmaTileMatmulConfig;
        type S = CmmaStageMatmulConfig<T>;
        type G = CmmaGlobalMatmulConfig<S>;
        type B = CmmaBatchMatmulConfig<G>;
        const PLANE_DIM: u32 = 32;

        macro_rules! matmul_test {
            (
                                                                        $test_name:ident,
                                                                        $problem:expr,
                                                                        $cube_dim:expr,
                                                                        $cube_count:expr,
                                                                        $stage_size:ty,
                                                                        $tile_matmul_type:ident,
                                                                        $advanded_config:expr
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

                    run_matmul_test::<
                        EG,
                        ES,
                        EA,
                        TileMatmul,
                        StageMatmul,
                        GlobalMatmul,
                        BatchMatmul,
                        R,
                    >(problem, $cube_dim, $cube_count, $advanded_config, device);
                }
            };
        }

        #[test]
        pub fn test_batch_matmul_b3x4_g300x300x300_s64x64x32() {
            matmul_test!(
                test_batch_matmul_b3x4_g300x300x300_s64x64x32,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(5, 5, 12),
                S64x64x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3x4_g300x300x300_s64x64x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3x4_g108x108x243_s64x64x32() {
            matmul_test!(
                test_batch_matmul_b3x4_g108x108x243_s64x64x32,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(2, 2, 1),
                S64x64x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3x4_g108x108x243_s64x64x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3x4_g256x256x256_s64x64x32() {
            matmul_test!(
                test_batch_matmul_b3x4_g256x256x256_s64x64x32,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(4, 4, 12),
                S64x64x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3x4_g256x256x256_s64x64x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3_g256x256x256_s64x64x32() {
            matmul_test!(
                test_batch_matmul_b3_g256x256x256_s64x64x32,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(4, 4, 3),
                S64x64x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3_g256x256x256_s64x64x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3_g16x16x16_s16x16x16_col_major() {
            matmul_test!(
                test_batch_matmul_b3_g16x16x16_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 3),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3_g16x16x16_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_b3_g16x16x16_s16x16x16() {
            matmul_test!(
                test_batch_matmul_b3_g16x16x16_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 3),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_b3_g16x16x16_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_g256x256x256_s64x64x32() {
            matmul_test!(
                test_batch_matmul_g256x256x256_s64x64x32,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(4, 4, 1),
                S64x64x32,
                $i_16x16x16,
                AdvancedConfig {
                    tiling_order: TilingOrderConfig::YMajor
                }
            );
            test_batch_matmul_g256x256x256_s64x64x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_g32x32x32_s16x16x16_col_y_major() {
            matmul_test!(
                test_batch_matmul_g32x32x32_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(2, 2, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig {
                    tiling_order: TilingOrderConfig::YMajor
                }
            );
            test_batch_matmul_g32x32x32_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_batch_matmul_g32x32x32_s16x16x16() {
            matmul_test!(
                test_batch_matmul_g32x32x32_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(2, 2, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g32x32x32_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_g16x14x16_s16x16x16_rhs_col_major() {
            matmul_test!(
                test_batch_matmul_g14x16x16_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g14x16x16_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_g16x12x16_s16x16x16() {
            matmul_test!(
                test_batch_matmul_g14x16x16_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g14x16x16_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_g16x16x12_s16x16x16() {
            matmul_test!(
                test_batch_matmul_g14x16x16_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_batch_matmul_g14x16x16_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g60x60x120_s64x64x32() {
            matmul_test!(
                test_global_matmul_g60x60x120_s64x64x32,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(1, 1, 1),
                S64x64x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_global_matmul_g60x60x120_s64x64x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x36_s16x16x16() {
            matmul_test!(
                test_global_matmul_g16x16x36_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x36_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g12x12x16_s16x16x16() {
            matmul_test!(
                test_global_matmul_g12x12x16_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_global_matmul_g12x12x16_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16_unlined() {
            matmul_test!(
                test_global_matmul_g16x16x16_s16x16x16_unlined,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s16x16x16_unlined::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16_line2() {
            matmul_test!(
                test_global_matmul_g16x16x16_s16x16x16_line2,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s16x16x16_line2::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16() {
            matmul_test!(
                test_global_matmul_g16x16x16_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s16x16x16::<TestRuntime>(&Default::default())
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x32,
                $i_16x16x16,
                AdvancedConfig {
                    tiling_order: TilingOrderConfig::YMajor
                }
            );

            test_global_matmul_ymajor::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x32_s16x16x16() {
            matmul_test!(
                test_global_matmul_g16x16x32_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x32_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16_col_major() {
            matmul_test!(
                test_global_matmul_g16x16x16_s16x16x16_col_major,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x16_s16x16x16_col_major::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x16x128_s16x16x16() {
            matmul_test!(
                test_global_matmul_g16x16x128_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x16x128_s16x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g32x16x128_s32x16x16() {
            matmul_test!(
                test_global_matmul_g32x16x128_s32x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g32x16x128_s32x16x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g32x32x224_s32x32x32() {
            matmul_test!(
                test_global_matmul_g32x32x224_s32x32x32,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g32x32x224_s32x32x32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g16x32x16_s16x32x16() {
            matmul_test!(
                test_global_matmul_g16x32x16_s16x32x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x32x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g16x32x16_s16x32x16::<TestRuntime>(&Default::default())
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_col_major_tiling::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_global_matmul_g32x32x16_s32x32x16() {
            matmul_test!(
                test_global_matmul_g32x32x16_s32x32x16,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );

            test_global_matmul_g32x32x16_s32x32x16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_stage_matmul_s16x32x16() {
            matmul_test!(
                test_stage_matmul_s16x32x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x32x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s16x32x16::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s16x16x16() {
            matmul_test!(
                test_stage_matmul_s16x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S16x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_row_col() {
            matmul_test!(
                test_stage_matmul_s32x32x32_row_col,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x32x32_row_col::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_col_row() {
            matmul_test!(
                test_stage_matmul_s32x32x32_col_row,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x32x32_col_row::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_col_col() {
            matmul_test!(
                test_stage_matmul_s32x32x32_col_col,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x32x32_col_col::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x16x16() {
            matmul_test!(
                test_stage_matmul_s32x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x16x16::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x8x16_col_major() {
            matmul_test!(
                test_stage_matmul_s32x8x16_col_major,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S32x8x16,
                $i_32x8x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x8x16_col_major::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s128x16x16() {
            matmul_test!(
                test_stage_matmul_s128x16x16,
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
                },
                CubeDim::new(PLANE_DIM, 8, 1),
                CubeCount::Static(1, 1, 1),
                S128x16x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s128x16x16::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s64x64x16() {
            matmul_test!(
                test_stage_matmul_s64x64x16,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(1, 1, 1),
                S64x64x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s64x64x16::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s64x64x32() {
            matmul_test!(
                test_stage_matmul_s64x64x32,
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
                },
                CubeDim::new(PLANE_DIM, 4, 1),
                CubeCount::Static(1, 1, 1),
                S64x64x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s64x64x32::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x32x16() {
            matmul_test!(
                test_stage_matmul_s32x32x16,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x16,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x32x16::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32() {
            matmul_test!(
                test_stage_matmul_s32x32x32,
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
                },
                CubeDim::new(PLANE_DIM, 2, 1),
                CubeCount::Static(1, 1, 1),
                S32x32x32,
                $i_16x16x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x32x32::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x8x16_row_major() {
            matmul_test!(
                test_stage_matmul_s32x8x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S32x8x16,
                $i_32x8x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s32x8x16::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s8x32x16() {
            matmul_test!(
                test_stage_matmul_s8x32x16,
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
                },
                CubeDim::new(PLANE_DIM, 1, 1),
                CubeCount::Static(1, 1, 1),
                S8x32x16,
                $i_8x32x16,
                AdvancedConfig::default()
            );
            test_stage_matmul_s8x32x16::<TestRuntime>(&Default::default());
        }
    };
}
