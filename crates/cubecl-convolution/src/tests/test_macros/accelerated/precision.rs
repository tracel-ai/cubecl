#[macro_export]
macro_rules! testgen_convolution_accelerated_precision {
    ($algorithm: ty) => {
        #[cfg(feature = "conv_tests_f16")]
        mod f16_ty {
            use super::*;

            $crate::testgen_convolution_accelerated_tiling_scheme!(
                $algorithm,
                (half::f16, half::f16)
            );
        }

        #[cfg(feature = "conv_tests_f32")]
        mod f32_ty {
            use super::*;

            $crate::testgen_convolution_accelerated_tiling_scheme!($algorithm, (f32, tf32));
        }
    };
}
