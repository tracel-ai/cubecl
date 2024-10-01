#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal_mock {
    () => {
        use cubecl_linalg::matmul::cmma::{CmmaMatmul, S16_16_16};
        use cubecl_linalg::matmul::dummy_unit_instruction::DummyUnitInstruction;
        use cubecl_linalg::matmul::tests;

        #[test]
        pub fn test_matmul_instruction_16_16_16() {
            tests::matmul_instruction::test_matmul_instruction::<
                DummyUnitInstruction<f32, f32>,
                f32,
                f32,
                TestRuntime,
            >(&Default::default())
        }
    };
}
