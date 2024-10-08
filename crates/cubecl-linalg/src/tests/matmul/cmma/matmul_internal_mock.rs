#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal_mock {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::{
            B128x128x16, B128x16x16, B16x16x16, B16x32x16, B32x16x16, B32x32x16, B32x32x32,
            B32x8x16, B64x64x16, B64x64x32, B8x32x16, CmmaBlockMatmul,
        };
        use cubecl_linalg::matmul::dummy_unit_instruction::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::problem::MatmulProblem;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::{
            test_fixed_matmul, test_tensor_matmul,
        };

        use cubecl_linalg::matmul::cube_matmul::base::CmmaCubeMatmul;
        use cubecl_linalg::matmul::tile_io::loading::smem::tiled_layout::{
            ColMajorTiling, RowMajorTiling,
        };
        use cubecl_linalg::matmul::tile_io::loading::tensor_loader::{
            LhsTensorLoader, RhsTensorLoader,
        };

        #[test]
        pub fn test_cube_matmul_s16x16x16_b16x16x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16x16x16>,
                    LhsTensorLoader<f32, RowMajorTiling>,
                    RhsTensorLoader<f32, RowMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 16, 16, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s16x16x16_b16x16x16_col_major() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16x16x16>,
                    LhsTensorLoader<f32, RowMajorTiling>,
                    RhsTensorLoader<f32, RowMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 16, 16, MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s16x16x128_b16x16x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16x16x16>,
                    LhsTensorLoader<f32, RowMajorTiling>,
                    RhsTensorLoader<f32, RowMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 16, 128, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s32x16x128_b32x16x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x16x16>,
                    LhsTensorLoader<f32, RowMajorTiling>,
                    RhsTensorLoader<f32, RowMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 16, 128, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s32x32x224_b32x32x32() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x32>,
                    LhsTensorLoader<f32, RowMajorTiling>,
                    RhsTensorLoader<f32, RowMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 32, 224, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s16x32x16_b16x32x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16x32x16>,
                    LhsTensorLoader<f32, RowMajorTiling>,
                    RhsTensorLoader<f32, RowMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 32, 16, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_col_major_tiling() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x32>,
                    LhsTensorLoader<f32, ColMajorTiling>,
                    RhsTensorLoader<f32, ColMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 32, 32, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_cube_matmul_s32x32x16_b32x32x16() {
            test_tensor_matmul::<
                CmmaCubeMatmul<
                    f32,
                    CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x16>,
                    LhsTensorLoader<f32, RowMajorTiling>,
                    RhsTensorLoader<f32, RowMajorTiling>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 32, 16, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b16x32x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16x32x16>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B16x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x32x32_row_col() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x32x32_col_row() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::ColMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x32x32_col_col() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_block_matmul_b32x16x16() {
            test_fixed_matmul::<
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x16x16>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B128x16x16>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B64x64x16>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B64x64x32>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x16>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, B32x32x32>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction32_8_16<f32, f32>, B32x8x16>,
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
                CmmaBlockMatmul<f32, f32, DummyUnitInstruction8_32_16<f32, f32>, B8x32x16>,
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
