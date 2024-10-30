#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tiling2d_matmul {
    () => {
        use super::*;

        #[test]
        pub fn test_matmul_tiling2d_one_cube() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_one_cube::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_tiling2d_several_cubes() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_several_cubes::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_tiling2d_with_check_bounds() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_with_check_bounds::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_tiling2d_with_batches() {
            cubecl_linalg::matmul::tests::tiling2d::test_matmul_tiling2d_with_batches::<TestRuntime>(
                &Default::default(),
            )
        }
    };
}
