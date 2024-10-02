#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal_mock {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::{
            CmmaMatmul, S128_128_16, S128_16_16, S16_16_16, S32_16_16, S32_32_16, S32_32_32,
            S32_8_16, S64_64_16, S8_32_16,
        };
        use cubecl_linalg::matmul::dummy_unit_instruction::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::tests;

        #[test]
        pub fn array_into_row_major_block_layout_test() {
            tests::dummy_tile::array_into_row_major_block_layout_test::<TestRuntime>(
                false,
                &Default::default(),
            )
        }

        #[test]
        pub fn array_into_row_major_block_layout_revert_test() {
            tests::dummy_tile::array_into_row_major_block_layout_test::<TestRuntime>(
                true,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_16_16_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16_16_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_block_matmul_32_16_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32_16_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_block_matmul_128_16_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S128_16_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_block_matmul_64_64_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S64_64_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_block_matmul_32_32_16() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32_32_16>,
                f32,
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn test_block_matmul_32_32_32() {
            tests::block_matmul::test_block_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32_32_32>,
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
