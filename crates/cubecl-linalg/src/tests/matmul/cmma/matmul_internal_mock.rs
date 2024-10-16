#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal_mock {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::config::CmmaConfig;
        use cubecl_linalg::matmul::cmma_matmul::global::CmmaGlobalMatmul;
        use cubecl_linalg::matmul::cmma_matmul::stage::{
            CmmaStageMatmul, S128x128x16, S128x16x16, S16x16x16, S16x16x32, S16x32x16, S32x16x16,
            S32x32x16, S32x32x32, S32x8x16, S64x64x16, S64x64x32, S8x32x16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::dummy::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::matmul_global::{
            LhsTensorLoader, RhsTensorLoader, TensorUnloader,
        };
        use cubecl_linalg::matmul::matmul_stage::{SharedMemoryStage, XMajorTiling, YMajorTiling};
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::problem::MatmulProblem;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_matmul;

        #[test]
        pub fn test_global_matmul_g60x60x120_s64x64x32() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S64x64x32,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    60,
                    60,
                    120,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                4,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x36_s16x16x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    36,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g12x12x16_s16x16x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    12,
                    12,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_precisions() {
            type EG = i32;
            type ES = i32;
            type EA = f32;
            type INSTR = DummyUnitInstruction16_16_16<ES, EA>;
            type STAGE = CmmaStageMatmul<ES, EG, EA, INSTR, S16x16x16>;
            type GLOBAL = CmmaGlobalMatmul<
                EG,
                ES,
                STAGE,
                LhsTensorLoader<EG, ES, SharedMemoryStage<ES, XMajorTiling>>,
                RhsTensorLoader<EG, ES, SharedMemoryStage<ES, XMajorTiling>>,
                TensorUnloader<EG>,
            >;
            test_matmul::<GLOBAL, EG, EG, TestRuntime>(
                // Can't accumulate more, it will fail because of i32
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16_unlined() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    1,
                    1,
                    1,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16_line2() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    2,
                    2,
                    2,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_ymajor() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S32x32x32,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, YMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, YMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    32,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x32_s16x16x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    64,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16_col_major() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::ColMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x128_s16x16x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    128,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g32x16x128_s32x16x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S32x16x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    16,
                    128,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g32x32x224_s32x32x32() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S32x32x32,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    224,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x32x16_s16x32x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S16x32x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    32,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_col_major_tiling() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S32x32x32,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    32,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g32x32x16_s32x32x16() {
            test_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    f32,
                    CmmaStageMatmul<
                        f32,
                        f32,
                        f32,
                        DummyUnitInstruction16_16_16<f32, f32>,
                        S32x32x16,
                    >,
                    LhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s16x32x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x32x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    32,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s16x16x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_row_col() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    32,
                    MatrixLayout::RowMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_col_row() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    32,
                    MatrixLayout::ColMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_col_col() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    32,
                    MatrixLayout::ColMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x16x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_i32x8x16_col_major() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction32_8_16<f32, f32>, S32x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    16,
                    16,
                    MatrixLayout::ColMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        #[ignore = "Should panic or not depending on line size"]
        // Line size too large gives out of bounds
        pub fn test_stage_matmul_s128x16x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S128x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    128,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                8,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s64x64x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S64x64x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    64,
                    64,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                4,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s64x64x32() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S64x64x32>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    64,
                    64,
                    32,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                4,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    32,
                    32,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                2,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x8x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction32_8_16<f32, f32>, S32x8x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    32,
                    8,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s8x32x16() {
            test_matmul::<
                CmmaStageMatmul<f32, f32, f32, DummyUnitInstruction8_32_16<f32, f32>, S8x32x16>,
                f32,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(
                    8,
                    32,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16() {
            test_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16_row_col() {
            test_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16_col_row() {
            test_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::ColMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16_col_col() {
            test_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    16,
                    16,
                    16,
                    MatrixLayout::ColMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_32x8x16() {
            test_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    32,
                    8,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_32x8x16_col_major() {
            test_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    32,
                    8,
                    16,
                    MatrixLayout::ColMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_8x32x16() {
            test_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    8,
                    32,
                    16,
                    MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_8x32x16_col_major() {
            test_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                MatmulProblem::new(
                    8,
                    32,
                    16,
                    MatrixLayout::ColMajor,
                    MatrixLayout::ColMajor,
                    4,
                    4,
                    4,
                ),
                1,
                &Default::default(),
            )
        }
    };
}
