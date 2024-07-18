#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul {
    () => {
        use cubecl_lac::matmul;

        use super::*;

        #[test]
        pub fn test_matmul_cmma_one_cube() {
            matmul::matmul_tests::test_matmul_cmma_one_cube::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_several_cubes() {
            matmul::matmul_tests::test_matmul_cmma_several_cubes::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_with_check_bounds() {
            matmul::matmul_tests::test_matmul_cmma_with_check_bounds::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_with_batches() {
            matmul::matmul_tests::test_matmul_cmma_with_batches::<TestRuntime>(&Default::default())
        }
    };
}
