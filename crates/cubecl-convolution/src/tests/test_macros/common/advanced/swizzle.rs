#[macro_export]
macro_rules! testgen_convolution_swizzle {
    ($algorithm: ty, $precision: ty, $selection_builder: expr) => {
        use cubecl_matmul::components::SwizzleConfig;
        use cubecl_matmul::components::stage::SwizzleMode;

        #[cfg(not(feature = "conv_tests_swizzle"))]
        $crate::testgen_convolution_partition_buffering!(
            $algorithm,
            $precision,
            $selection_builder.shared_swizzle(SwizzleConfig::default())
        );

        #[cfg(feature = "conv_tests_swizzle")]
        mod none {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $precision,
                $selection_builder.shared_swizzle(SwizzleConfig {
                    lhs: SwizzleMode::None,
                    rhs: SwizzleMode::None,
                    ..Default::default()
                })
            );
        }

        #[cfg(feature = "conv_tests_swizzle")]
        mod b32 {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $precision,
                $selection_builder.shared_swizzle(SwizzleConfig {
                    lhs: SwizzleMode::B32,
                    rhs: SwizzleMode::B32,
                    ..Default::default()
                })
            );
        }

        #[cfg(feature = "conv_tests_swizzle")]
        mod b64 {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $precision,
                $selection_builder.shared_swizzle(SwizzleConfig {
                    lhs: SwizzleMode::B64,
                    rhs: SwizzleMode::B64,
                    ..Default::default()
                })
            );
        }

        #[cfg(feature = "conv_tests_swizzle")]
        mod b128 {
            use super::*;

            $crate::testgen_convolution_partition_buffering!(
                $algorithm,
                $precision,
                $selection_builder.shared_swizzle(SwizzleConfig {
                    lhs: SwizzleMode::B128,
                    rhs: SwizzleMode::B128,
                    ..Default::default()
                })
            );
        }
    };
}
