#[macro_export]
macro_rules! testgen_matmul_tma_stage {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr) => {
        use $crate::components::{LoadSpecializationConfig, StageSize};

        mod s1x1x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 1, n: 1, k: 1 },
                LoadSpecializationConfig::default()
            );
        }

        mod s2x2x1 {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Tma,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                StageSize { m: 2, n: 2, k: 1 },
                LoadSpecializationConfig::default()
            );
        }
    };
}
