#[macro_export]
macro_rules! testgen_matmul_launch {
    (Normal, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
        use super::*;
        use $crate::tests::layered::matmul_test_launcher::test_matmul_algorithm;

        #[test]
        pub fn test() {
            let client = TestRuntime::client(&Default::default());
            test_matmul_algorithm::<$algorithm, $precision, TestRuntime>(
                client, $problem, $selection,
            );
        }
    };

    (Tma, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
        use super::*;
        use $crate::tests::layered::tma_test_launcher::test_tma_matmul_algorithm;

        #[test]
        pub fn test() {
            let client = TestRuntime::client(&Default::default());
            test_tma_matmul_algorithm::<$algorithm, $precision, TestRuntime>(
                client, $problem, $selection,
            );
        }
    };
}
