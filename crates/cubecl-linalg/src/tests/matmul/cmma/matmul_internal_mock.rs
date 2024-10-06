#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal_mock {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::{
            CmmaBlockMatmul, B128_128_16, B128_16_16, B16_16_16, B16_32_16, B32_16_16, B32_32_16,
            B32_32_32, B32_8_16, B64_64_16, B64_64_32, B8_32_16,
        };
        use cubecl_linalg::matmul::dummy_unit_instruction::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::{
            test_fixed_matmul, test_tensor_matmul,
        };

        use cubecl_linalg::matmul::cube_matmul::base::CmmaCubeMatmul;

        #[test]
        pub fn test_cube_matmul_s16x16x128_b16x16x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16_16_16>,
                >,
                f32,
                TestRuntime,
            >(
                16,
                16,
                128,
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s32x16x128_b32x16x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32_16_16>,
                >,
                f32,
                TestRuntime,
            >(
                32,
                16,
                128,
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s16x32x16_b16x32x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16_32_16>,
                >,
                f32,
                TestRuntime,
            >(
                16,
                32,
                16,
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b16x32x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16_32_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b16x16x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16_16_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x16x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32_16_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b128x16x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B128_16_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b64x64x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B64_64_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b64x64x32() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B64_64_32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x32x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32_32_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x32x32() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32_32_32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x8x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction32_8_16<f32, f32>, B32_8_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b8x32x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction8_32_16<f32, f32>, B8_32_16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_16x16x16() {
            test_fixed_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_16x16x16_col_major() {
            test_fixed_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_32x8x16() {
            test_fixed_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_32x8x16_col_major() {
            test_fixed_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_8x32x16() {
            test_fixed_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_fixed_matmul_instruction_8x32x16_col_major() {
            test_fixed_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }
    };
}
