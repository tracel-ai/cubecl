#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_instruction_cmma_f16 {
    () => {
        use cubecl_linalg::matmul::cmma::CmmaInstruction;
        use cubecl_linalg::matmul::tests;
        use half::{bf16, f16};

        #[test]
        pub fn test_matmul_instruction_f16_in_f16_out() {
            tests::matmul_instruction::test_matmul_instruction::<
                CmmaInstruction<f16, f16>,
                f16,
                f16,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_instruction_f16_in_f32_out() {
            tests::matmul_instruction::test_matmul_instruction::<
                CmmaInstruction<f16, f32>,
                f16,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_instruction_bf16_in_f32_out() {
            tests::matmul_instruction::test_matmul_instruction::<
                CmmaInstruction<bf16, f32>,
                bf16,
                f32,
                TestRuntime,
            >(&Default::default())
        }
    };
}
