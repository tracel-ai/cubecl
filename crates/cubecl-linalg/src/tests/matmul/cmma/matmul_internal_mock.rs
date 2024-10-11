#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal_mock {
    () => {
        use cubecl_linalg::matmul::cmma_matmul::global::CmmaGlobalMatmul;
        use cubecl_linalg::matmul::cmma_matmul::stage::{
            CmmaStageMatmul, S128x128x16, S128x16x16, S16x16x16, S16x32x16, S32x16x16, S32x32x16,
            S32x32x32, S32x8x16, S64x64x16, S64x64x32, S8x32x16,
        };
        use cubecl_linalg::matmul::cmma_matmul::tile::dummy::{
            DummyUnitInstruction16_16_16, DummyUnitInstruction32_8_16, DummyUnitInstruction8_32_16,
        };
        use cubecl_linalg::matmul::matmul_global::{
            LhsTensorLoader, RhsTensorLoader, TensorUnloader,
        };
        use cubecl_linalg::matmul::matmul_stage::{SharedMemoryStage, XMajorTiling, YMajorTiling};
        use cubecl_linalg::matmul::matrix_layout::MatrixLayout;
        use cubecl_linalg::matmul::problem::MatmulProblem;
        use cubecl_linalg::matmul::tests::matmul_test_launcher::{
            test_fixed_matmul, test_tensor_matmul,
        };

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x16x16>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 16, 16, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x32_s16x16x16() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x16x16>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 16, 64, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x16_s16x16x16_col_major() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x16x16>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 16, 16, MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x16x128_s16x16x16() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x16x16>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 16, 128, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g32x16x128_s32x16x16() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x16x16>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 16, 128, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g32x32x224_s32x32x32() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 32, 224, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g16x32x16_s16x32x16() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x32x16>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(16, 32, 16, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_col_major_tiling() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 32, 32, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_g32x32x16_s32x32x16() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x16>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 32, 16, MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s16x32x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x32x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s16x16x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S16x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_row_col() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_col_row() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::ColMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32_col_col() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x16x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_i32x8x16_col_major() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction32_8_16<f32, f32>, S32x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_global_matmul_complicated() {
            test_tensor_matmul::<
                CmmaGlobalMatmul<
                    f32,
                    CmmaStageMatmul<f32, f32, DummyUnitInstruction32_8_16<f32, f32>, S32x32x32>,
                    LhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    RhsTensorLoader<f32, SharedMemoryStage<f32, XMajorTiling>>,
                    TensorUnloader<f32>,
                >,
                f32,
                TestRuntime,
            >(
                MatmulProblem::new(32, 32, 128, MatrixLayout::ColMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s128x16x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S128x16x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s64x64x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S64x64x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s64x64x32() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S64x64x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x32x32() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction16_16_16<f32, f32>, S32x32x32>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s32x8x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction32_8_16<f32, f32>, S32x8x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_stage_matmul_s8x32x16() {
            test_fixed_matmul::<
                CmmaStageMatmul<f32, f32, DummyUnitInstruction8_32_16<f32, f32>, S8x32x16>,
                f32,
                f32,
                TestRuntime,
            >(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16() {
            test_fixed_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16_row_col() {
            test_fixed_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16_col_row() {
            test_fixed_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_16x16x16_col_col() {
            test_fixed_matmul::<DummyUnitInstruction16_16_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_32x8x16() {
            test_fixed_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_32x8x16_col_major() {
            test_fixed_matmul::<DummyUnitInstruction32_8_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_8x32x16() {
            test_fixed_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
                &Default::default(),
            )
        }

        #[test]
        pub fn test_tile_matmul_8x32x16_col_major() {
            test_fixed_matmul::<DummyUnitInstruction8_32_16<f32, f32>, f32, f32, TestRuntime>(
                (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
                &Default::default(),
            )
        }
    };
}
