#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal {
    () => {
        use cubecl_linalg::matmul::cmma_instruction::base::{
            CmmaInstruction16_16_16, CmmaInstruction32_8_16, CmmaInstruction8_32_16,
        };
        use cubecl_linalg::matmul::cmma_matmul::{CmmaBlockMatmul, S16_16_16, S32_8_16, S8_32_16};
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_fixed_matmul;
        use half::{bf16, f16};

        #[test]
        pub fn test_block_matmul_16_16_16_f32() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f16, f32, CmmaInstruction16_16_16<f16, f32>, S16_16_16>,
                f16,
                f16,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_16_16_16_f16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f16, f16, CmmaInstruction16_16_16<f16, f16>, S16_16_16>,
                f16,
                f16,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_32_8_16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f16, f32, CmmaInstruction32_8_16<f16, f32>, S32_8_16>,
                f16,
                f16,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_8_32_16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f16, f32, CmmaInstruction8_32_16<f16, f32>, S8_32_16>,
                f16,
                f16,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_f16_in_f16_out() {
            test_fixed_matmul::<CmmaInstruction16_16_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_f16_in_f16_out_col_major() {
            test_fixed_matmul::<CmmaInstruction16_16_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_f16_in_f32_out() {
            test_fixed_matmul::<CmmaInstruction16_16_16<f16, f32>, f16, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        #[ignore]
        pub fn test_fixed_matmul_instruction_bf16_in_f32_out() {
            test_fixed_matmul::<CmmaInstruction16_16_16<bf16, f32>, bf16, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_32_8_16() {
            test_fixed_matmul::<CmmaInstruction32_8_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_8_32_16() {
            test_fixed_matmul::<CmmaInstruction8_32_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_8_32_16_col_major() {
            test_fixed_matmul::<CmmaInstruction8_32_16<f16, f16>, f16, f16, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }
    };
}
