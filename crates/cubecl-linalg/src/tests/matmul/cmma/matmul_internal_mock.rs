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
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::tests::dummy_tile;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::test_matmul;

        #[test]
        pub fn array_into_row_major_block_layout_test() {
            dummy_tile::array_into_row_major_block_layout_test::<TestRuntime>(
                false,
                &Default::default(),
            )
        }

        #[test]
        pub fn array_into_row_major_block_layout_revert_test() {
            dummy_tile::array_into_row_major_block_layout_test::<TestRuntime>(
                true,
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_16_16_16() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16_16_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_32_16_16() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32_16_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_128_16_16() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S128_16_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_64_64_16() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S64_64_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_32_32_16() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32_32_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_32_32_32() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32_32_32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_32_8_16() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction32_8_16<f32, f32>, S32_8_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_8_32_16() {
            test_matmul::<
                CmmaMatmul<f32, f32, DummyUnitInstruction8_32_16<f32, f32>, S8_32_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_instruction_16_16_16() {
            test_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_instruction_16_16_16_col_major() {
            test_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_instruction_32_8_16() {
            test_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_instruction_32_8_16_col_major() {
            test_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_instruction_8_32_16() {
            test_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_instruction_8_32_16_col_major() {
            test_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }
    };
}
