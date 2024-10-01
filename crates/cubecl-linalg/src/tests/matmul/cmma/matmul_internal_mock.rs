#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal_mock {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::{CmmaMatmul, S16_16_16, S32_8_16, S8_32_16};
        use cubecl_linalg::matmul::dummy_unit_instruction::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::tests;

        #[test]
        pub fn test_block_matmul_16_16_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16_16_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_block_matmul_32_8_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction32_8_16<f32, f32>, S32_8_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_block_matmul_8_32_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction8_32_16<f32, f32>, S8_32_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_instruction_16_16_16() {
            tests::matmul_instruction::test_matmul_instruction::<
                DummyUnitInstruction16_16_16<f32, f32>,
                f32,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_instruction_32_8_16() {
            tests::matmul_instruction::test_matmul_instruction::<
                DummyUnitInstruction32_8_16<f32, f32>,
                f32,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_matmul_instruction_8_32_16() {
            tests::matmul_instruction::test_matmul_instruction::<
                DummyUnitInstruction8_32_16<f32, f32>,
                f32,
                f32,
                TestRuntime,
            >(&Default::default())
        }
    };
}
