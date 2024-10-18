#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::global::CmmaGlobalMatmul;
        use cubecl_linalg::matmul::cmma_matmul::global::CmmaGlobalMatmulConfig;
        use cubecl_linalg::matmul::cmma_matmul::global::{
            LhsTensorLoader, RhsTensorLoader, TensorUnloader,
        };
        use cubecl_linalg::matmul::cmma_matmul::stage::CmmaStageMatmulConfig;
        use cubecl_linalg::matmul::cmma_matmul::stage::{
            CmmaStageMatmul, S128x128x16, S128x16x16, S16x16x16, S16x16x32, S16x32x16, S32x16x16,
            S32x32x16, S32x32x32, S32x8x16, S64x64x16, S64x64x32, S8x32x16, SharedMemoryStage,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::dummy::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::CmmaTileMatmulConfig;
        use cubecl_linalg::matmul::matmul_stage::StageMatmul;
        use cubecl_linalg::matmul::matmul_stage::{XMajorTiling, YMajorTiling};
        use cubecl_linalg::matmul::matmul_tile::TileMatmul;
        use cubecl_linalg::matmul::matrix::MatrixLayout;
        use cubecl_linalg::matmul::problem::MatmulProblem;
        use cubecl_linalg::matmul::tests::create_stage_dim;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_matmul;
        use cubecl_linalg::matmul::tests::run_matmul_test;
        use cubecl_linalg::matmul::stage_dim::StageDim;

        type T = CmmaTileMatmulConfig;
        type S = CmmaStageMatmulConfig<T>;
        type G = CmmaGlobalMatmulConfig<S>;

        macro_rules! matmul_test {
            (
                                                        $test_name:ident,
                                                        $problem:expr,
                                                        $num_planes:expr,
                                                        $eg:ty, $es:ty, $ea:ty, $stage_size:ty,
                                                        $tile_matmul_type:ident
                                                    ) => {
                pub fn $test_name<R: Runtime>(device: &R::Device) {
                    type T = CmmaTileMatmulConfig;
                    type S = CmmaStageMatmulConfig<T>;
                    type G = CmmaGlobalMatmulConfig<S>;

                    let problem = $problem;

                    let num_planes = $num_planes;
                    type EG = $eg;
                    type ES = $es;
                    type EA = $ea;
                    type StageSize = $stage_size;

                    type TileMatmul = $tile_matmul_type<ES, EA, T>;
                    type StageMatmul = CmmaStageMatmul<ES, EG, EA, TileMatmul, StageSize, S>;
                    type GlobalMatmul = CmmaGlobalMatmul<EG, ES, StageMatmul, G>;

                    run_matmul_test::<EG, ES, EA, TileMatmul, StageMatmul, GlobalMatmul, R>(
                        problem, num_planes, device,
                    );
                }
            };
        }

        #[test]
        pub fn test_global_matmul_g60x60x120_s64x64x32() {
            matmul_test!(
                test_global_matmul_g60x60x120_s64x64x32,
                MatmulProblem {
                    m: 60,
                    n: 60,
                    k: 120,
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                4,
                f32,
                f32,
                f32,
                S64x64x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 1,
                    rhs_line_size: 1,
                    out_line_size: 1,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 2,
                    rhs_line_size: 2,
                    out_line_size: 2,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x32x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x32x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S16x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x16x16,
                DummyUnitInstruction16_16_16
            );
            test_stage_matmul_s32x16x16::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_i32x8x16_col_major() {
            matmul_test!(
                test_stage_matmul_i32x8x16_col_major,
                MatmulProblem {
                    m: 32,
                    n: 16,
                    k: 16,
                    lhs_layout: MatrixLayout::ColMajor,
                    rhs_layout: MatrixLayout::ColMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S32x16x16,
                DummyUnitInstruction32_8_16
            );
            test_stage_matmul_i32x8x16_col_major::<TestRuntime>(&Default::default());
        }

        #[test]
        #[ignore]
        pub fn test_stage_matmul_s128x16x16() {
            matmul_test!(
                test_stage_matmul_s128x16x16,
                MatmulProblem {
                    m: 128,
                    n: 16,
                    k: 16,
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                8,
                f32,
                f32,
                f32,
                S128x16x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                4,
                f32,
                f32,
                f32,
                S64x64x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                4,
                f32,
                f32,
                f32,
                S64x64x32,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x16,
                DummyUnitInstruction16_16_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                2,
                f32,
                f32,
                f32,
                S32x32x32,
                DummyUnitInstruction16_16_16
            );
            test_stage_matmul_s32x32x32::<TestRuntime>(&Default::default());
        }

        #[test]
        pub fn test_stage_matmul_s32x8x16() {
            matmul_test!(
                test_stage_matmul_s32x8x16,
                MatmulProblem {
                    m: 32,
                    n: 8,
                    k: 16,
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S32x8x16,
                DummyUnitInstruction32_8_16
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
                    lhs_layout: MatrixLayout::RowMajor,
                    rhs_layout: MatrixLayout::RowMajor,
                    lhs_line_size: 4,
                    rhs_line_size: 4,
                    out_line_size: 4,
                },
                1,
                f32,
                f32,
                f32,
                S8x32x16,
                DummyUnitInstruction8_32_16
            );
            test_stage_matmul_s8x32x16::<TestRuntime>(&Default::default());
        }
    };
}
