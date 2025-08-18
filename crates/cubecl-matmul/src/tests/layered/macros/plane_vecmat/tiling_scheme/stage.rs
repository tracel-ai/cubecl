#[macro_export]
macro_rules! testgen_matmul_plane_vecmat_stage {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::StageSize;

        mod s1x1x1 {
            use super::*;

            $crate::testgen_matmul_advanced!(
                Normal,
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 1, n: 1, k: 1 })
            );
        }

        mod s1x2x1 {
            use super::*;

            $crate::testgen_matmul_advanced!(
                Normal,
                $algorithm,
                $precision,
                $tiling_scheme_builder.with_stage_size(StageSize { m: 1, n: 2, k: 1 })
            );
        }
    };
}
