#[macro_export]
macro_rules! testgen_matmul_tma_stage {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::StageSize;

        mod s1x1x1 {
            use super::*;

            $crate::testgen_matmul_advanced!(
                Tma,
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 1, n: 1, k: 1 })
            );
        }

        mod s2x2x1 {
            use super::*;

            $crate::testgen_matmul_advanced!(
                Tma,
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 2, n: 2, k: 1 })
            );
        }
    };
}
