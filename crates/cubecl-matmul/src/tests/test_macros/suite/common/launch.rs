#[macro_export]
macro_rules! testgen_matmul_launch {
    (PlaneAccelerated, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
        use super::*;
        use $crate::tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm;

        #[test]
        pub fn test() {
            let client = TestRuntime::client(&Default::default());
            test_matmul_algorithm::<$algorithm, $precision, TestRuntime>(
                client, $problem, $selection,
            );
        }
    };

    (Unit, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
        use super::*;
        use $crate::tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm;

        #[test]
        pub fn test() {
            let client = TestRuntime::client(&Default::default());
            test_matmul_algorithm::<$algorithm, $precision, TestRuntime>(
                client, $problem, $selection,
            );
        }
    };

    (Tma, $algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr, $stage_size: expr, $specialized: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_matmul::tests::test_macros::suite::tma::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >($selection, $problem);
        }
    };

    (Quantized, $algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr, $stage_size: expr, $specialized: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_matmul::tests::test_macros::suite::plane_accelerated::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >($selection, $problem);
        }
    };
}
