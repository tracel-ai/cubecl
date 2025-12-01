#[macro_export]
macro_rules! testgen_convolution_accelerated_stage {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use cubecl_matmul::components::StageSize;

        mod s1x1x1 {
            use super::*;

            $crate::testgen_convolution_advanced!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 1, n: 1, k: 1 })
            );
        }

        mod s2x2x1 {
            use super::*;

            $crate::testgen_convolution_advanced!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 2, n: 2, k: 1 })
            );
        }

        mod s4x4x1 {
            use super::*;

            $crate::testgen_convolution_advanced!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 4, n: 4, k: 1 })
            );
        }

        mod s8x4x1 {
            use super::*;

            $crate::testgen_convolution_advanced!(
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 8, n: 4, k: 1 })
            );
        }
    };
}
