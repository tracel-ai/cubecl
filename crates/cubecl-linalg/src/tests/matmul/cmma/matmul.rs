#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_matmul {
    () => {
        #[test]
        pub fn test_matmul_cmma_all() {
            tests::cmma::table_test::test_cmma_all::<TestRuntime>(&Default::default())
        }
    };
}
