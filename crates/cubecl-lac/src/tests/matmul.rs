#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul {
    () => {
        use cubecl_lac::matmul;

        use super::*;

        #[test]
        pub fn test_matmul_cmma_1() {
            matmul::matmul_tests::test_matmul_cmma_1::<TestRuntime>(&Default::default())
        }
    };
}
