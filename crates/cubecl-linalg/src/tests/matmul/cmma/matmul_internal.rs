#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::stage::{
            CmmaStageMatmul, S16x16x16, S32x8x16, S8x32x16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::base::{
            CmmaInstruction16_16_16, CmmaInstruction32_8_16, CmmaInstruction8_32_16,
        };
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_fixed_matmul;
        use half::{bf16, f16};

        #[test]
        pub fn test_stage_matmul_s16x16x16_f32() {
            test_fixed_matmul::<
                CmmaStageMatmul<f16, f32, f32, CmmaInstruction16_16_16<f16, f32>, S16x16x16>,
                f16,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s16x16x16_f16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f16, f16, f16, CmmaInstruction16_16_16<f16, f16>, S16x16x16>,
                f16,
                f16,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x8x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f16, f32, f32, CmmaInstruction32_8_16<f16, f32>, S32x8x16>,
                f16,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s8x32x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f16, f32, f32, CmmaInstruction8_32_16<f16, f32>, S8x32x16>,
                f16,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_f16_in_f16_out() {
            test_fixed_matmul::<CmmaInstruction16_16_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_f16_in_f16_out_col_major() {
            test_fixed_matmul::<CmmaInstruction16_16_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_f16_in_f32_out() {
            test_fixed_matmul::<CmmaInstruction16_16_16<f16, f32>, f16, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        #[ignore]
        pub fn test_fixed_matmul_instruction_bf16_in_f32_out() {
            test_fixed_matmul::<CmmaInstruction16_16_16<bf16, f32>, bf16, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_32_8_16() {
            test_fixed_matmul::<CmmaInstruction32_8_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_8_32_16() {
            test_fixed_matmul::<CmmaInstruction8_32_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_8_32_16_col_major() {
            test_fixed_matmul::<CmmaInstruction8_32_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                (4, 4, 4),
                Default::default(),
                &Default::default(),
            )
        }
    };
}
