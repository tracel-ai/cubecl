#[macro_export]
macro_rules! testgen_matmul_launch {
    (PlaneAccelerated, $algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr, $partition_count: expr, $stage_k: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::plane_accelerated::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >(
                $layouts,
                $tile,
                $partition_shape,
                $partition_count,
                $stage_k,
                $problem,
            );
        }
    };

    (Unit, $algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr, $partition_count: expr, $stage_k: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::unit::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >(
                $layouts,
                $tile,
                $partition_shape,
                $partition_count,
                $stage_k,
                $problem,
            );
        }
    };

    (Tma, $algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr, $partition_count: expr, $stage_k: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::tma::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >(
                $layouts,
                $tile,
                $partition_shape,
                $partition_count,
                $stage_k,
                $problem,
            );
        }
    };

    (Quantized, $algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr, $partition_count: expr, $stage_k: expr, $layouts: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::plane_accelerated::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >(
                $layouts,
                $tile,
                $partition_shape,
                $partition_count,
                $stage_k,
                $problem,
            );
        }
    };
}
