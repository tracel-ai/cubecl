#[macro_export]
macro_rules! testgen_matmul_launch {
    (PlaneAccelerated, $algorithm: ty, $precision: ty, $tile: expr, $partition: expr, $stage: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::plane_accelerated::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >($layouts, $tile, $partition, $stage, $problem);
        }
    };

    (Unit, $algorithm: ty, $precision: ty, $tile: expr, $partition: expr, $stage: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::unit::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >($layouts, $tile, $partition, $stage, $problem);
        }
    };

    (Tma, $algorithm: ty, $precision: ty, $tile: expr, $partition: expr, $stage: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::tma::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >($layouts, $tile, $stage, $problem);
        }
    };

    (Quantized, $algorithm: ty, $precision: ty, $tile: expr, $partition: expr, $stage: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::plane_accelerated::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >($layouts, $tile, $partition, $stage, $problem);
        }
    };
}
