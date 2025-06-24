#[macro_export]
macro_rules! testgen_matmul_launch {
    (PlaneAccelerated, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
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

    (Unit, $algorithm: ty, $precision: ty, ) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_matmul::tests::test_macros::suite::unit::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >($selection, $problem);
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
