#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::config::CmmaConfig;
        use cubecl_linalg::matmul::cmma_matmul::stage::{
            CmmaStageMatmul, S16x16x16, S32x8x16, S8x32x16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::base::{
            CmmaInstruction16_16_16, CmmaInstruction32_8_16, CmmaInstruction8_32_16,
        };
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::problem::MatmulProblem;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_matmul;
        use half::{bf16, f16};

        use cubecl_linalg::matmul::cmma_matmul::global::CmmaGlobalMatmul;
        use cubecl_linalg::matmul::cmma_matmul::tile::dummy::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::matmul_global::{
            LhsTensorLoader, RhsTensorLoader, TensorUnloader,
        };
        use cubecl_linalg::matmul::matmul_stage::{SharedMemoryStage, XMajorTiling, YMajorTiling};

        #[test]
        #[ignore]
        pub fn test_global_matmul_precisions() {
            type EG = f32;
            type ES = f16;
            type EA = f16;
            type INSTR = CmmaInstruction16_16_16<ES, EA>;
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
        pub fn test_stage_matmul_s16x16x16_f32() {
            test_matmul::<
                CmmaStageMatmul<f16, f32, f32, CmmaInstruction16_16_16<f16, f32>, S16x16x16>,
                f16,
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
        pub fn test_stage_matmul_s16x16x16_f16() {
            test_matmul::<
                CmmaStageMatmul<f16, f16, f16, CmmaInstruction16_16_16<f16, f16>, S16x16x16>,
                f16,
                f16,
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
        pub fn test_stage_matmul_s32x8x16() {
            test_matmul::<
                CmmaStageMatmul<f16, f32, f32, CmmaInstruction32_8_16<f16, f32>, S32x8x16>,
                f16,
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
                CmmaStageMatmul<f16, f32, f32, CmmaInstruction8_32_16<f16, f32>, S8x32x16>,
                f16,
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
        pub fn test_matmul_instruction_f16_in_f16_out() {
            test_matmul::<CmmaInstruction16_16_16<f16, f16>, f16, f16, TestRuntime>(
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
        pub fn test_matmul_instruction_f16_in_f16_out_col_major() {
            test_matmul::<CmmaInstruction16_16_16<f16, f16>, f16, f16, TestRuntime>(
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
        pub fn test_matmul_instruction_f16_in_f32_out() {
            test_matmul::<CmmaInstruction16_16_16<f16, f32>, f16, f32, TestRuntime>(
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
        #[ignore]
        pub fn test_matmul_instruction_bf16_in_f32_out() {
            test_matmul::<CmmaInstruction16_16_16<bf16, f32>, bf16, f32, TestRuntime>(
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
        pub fn test_matmul_instruction_32_8_16() {
            test_matmul::<CmmaInstruction32_8_16<f16, f16>, f16, f16, TestRuntime>(
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
        pub fn test_matmul_instruction_8_32_16() {
            test_matmul::<CmmaInstruction8_32_16<f16, f16>, f16, f16, TestRuntime>(
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
        pub fn test_matmul_instruction_8_32_16_col_major() {
            test_matmul::<CmmaInstruction8_32_16<f16, f16>, f16, f16, TestRuntime>(
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
