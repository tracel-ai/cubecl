#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tiling2d_matmul {
    () => {
        use super::*;

        #[test]
        pub fn test_matmul_tiling2d_one_cube() {
            tests::matmul_tests::test_matmul_tiling2d_one_cube::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_tiling2d_several_cubes() {
            tests::matmul_tests::test_matmul_tiling2d_several_cubes::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_tiling2d_with_check_bounds() {
            tests::matmul_tests::test_matmul_tiling2d_with_check_bounds::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_tiling2d_with_batches() {
            tests::matmul_tests::test_matmul_tiling2d_with_batches::<TestRuntime>(
                &Default::default(),
            )
        }
    };
}
