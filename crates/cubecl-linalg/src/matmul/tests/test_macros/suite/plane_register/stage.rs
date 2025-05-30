#[macro_export]
macro_rules! testgen_matmul_plane_register_stage {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr) => {
        use $crate::matmul::components::StageSize;

        mod s1x1x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Unit,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 1, n: 1, k: 1 }
            );
        }

        mod s2x1x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Unit,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 2, n: 1, k: 1 }
            );
        }
    };
}
