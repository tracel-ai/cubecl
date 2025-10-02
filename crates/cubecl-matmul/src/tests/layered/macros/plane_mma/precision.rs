#[macro_export]
macro_rules! testgen_matmul_mma_precision {
    ($algorithm: ty) => {
        #[cfg(feature = "matmul_tests_f16")]
        mod f16_ty {
            use super::*;

            $crate::testgen_matmul_mma_tiling_scheme!($algorithm, (half::f16, half::f16));
        }

        #[cfg(feature = "matmul_tests_f32")]
        mod f32_ty {
            use super::*;

            $crate::testgen_matmul_mma_tiling_scheme!($algorithm, (f32, f32));
        }
    };
}
