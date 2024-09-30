#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_instruction_cmma_f16 {
    () => {
        use cubecl_linalg::matmul::cmma::CmmaInstruction;
        use cubecl_linalg::matmul::tests;
        use half::f16;

        #[test]
        pub fn test_matmul_instruction_f16_input() {
            tests::matmul_instruction::test_matmul_instruction::<
                CmmaInstruction<f16, f16>,
                f16,
                f16,
                TestRuntime,
            >(&Default::default())
        }
    };
}
