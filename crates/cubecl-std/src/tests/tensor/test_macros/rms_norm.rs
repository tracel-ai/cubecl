#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_tensor_rms_norm {
    () => {
        mod rms_norm {
            $crate::testgen_tensor_rms_norm!(f32);
        }
    };
    ($float:ident) => {
        use super::*;
        use $crate::tests::tensor::rms_norm::test_rms_norm;

        pub type FloatT = $float;

        #[test]
        fn test_rms_norm_no_bias_small() {
            test_rms_norm::<TestRuntime, FloatT>(&Default::default(), &[2, 8], 1e-5, false);
        }

        #[test]
        fn test_rms_norm_small_with_bias() {
            test_rms_norm::<TestRuntime, FloatT>(&Default::default(), &[3, 4, 16], 1e-5, true);
        }

        #[test]
        fn test_rms_norm_prime_axis() {
            test_rms_norm::<TestRuntime, FloatT>(&Default::default(), &[4, 7], 1e-5, false);
        }

        #[test]
        fn test_rms_norm_large_row_no_bias() {
            test_rms_norm::<TestRuntime, FloatT>(
                &Default::default(),
                &[32, 2048, 4096],
                1e-5,
                false,
            );
        }

        #[test]
        fn test_rms_norm_large_row_with_bias() {
            test_rms_norm::<TestRuntime, FloatT>(
                &Default::default(),
                &[32, 2048, 4096],
                1e-5,
                true,
            );
        }

        #[test]
        fn test_rms_norm_mid_depth_no_bias() {
            test_rms_norm::<TestRuntime, FloatT>(
                &Default::default(),
                &[4, 4, 128],
                1e-5,
                false,
            );
        }

        #[test]
        fn test_rms_norm_mid_depth_with_bias() {
            test_rms_norm::<TestRuntime, FloatT>(
                &Default::default(),
                &[4, 4, 128],
                1e-5,
                true,
            );
        }

        #[test]
        fn test_rms_norm_deep_vector_no_bias() {
            test_rms_norm::<TestRuntime, FloatT>(
                &Default::default(),
                &[1, 1, 8192],
                1e-5,
                false,
            );
        }

        #[test]
        fn test_rms_norm_deep_vector_with_bias() {
            test_rms_norm::<TestRuntime, FloatT>(
                &Default::default(),
                &[1, 1, 8192],
                1e-5,
                true,
            );
        }
    };
    ([$($float:ident),*]) => {
        mod rms_norm {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;
                    $crate::testgen_tensor_rms_norm!($float);
                })*
            }
        }
    };
}
