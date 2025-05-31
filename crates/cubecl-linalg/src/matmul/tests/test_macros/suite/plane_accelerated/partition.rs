#[macro_export]
macro_rules! testgen_matmul_accelerated_partition_shape {
    ($algorithm: ty, $precision: ty, $tile: expr) => {
        use $crate::matmul::components::stage::TilesPerPartition;

        mod ps1x1 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_count!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 1, n: 1 }
            );
        }

        mod ps1x4 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_count!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 1, n: 4 }
            );
        }

        mod ps2x1 {
            use super::*;

            $crate::testgen_matmul_accelerated_partition_count!(
                $algorithm,
                $precision,
                $tile,
                TilesPerPartition { m: 2, n: 1 }
            );
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_accelerated_partition_count {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_shape: expr) => {
        use $crate::matmul::components::stage::PartitionsPerStage;

        mod pc1x1 {
            use super::*;

            $crate::testgen_matmul_stage_k!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                PartitionsPerStage { m: 1, n: 1 }
            );
        }

        mod pc2x2 {
            use super::*;

            $crate::testgen_matmul_stage_k!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                PartitionsPerStage { m: 2, n: 2 }
            );
        }

        mod pc4x1 {
            use super::*;

            $crate::testgen_matmul_stage_k!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                PartitionsPerStage { m: 4, n: 1 }
            );
        }

        mod pc4x4 {
            use super::*;

            $crate::testgen_matmul_stage_k!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                PartitionsPerStage { m: 4, n: 4 }
            );
        }

        mod pc8x4 {
            use super::*;

            $crate::testgen_matmul_stage_k!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                PartitionsPerStage { m: 8, n: 4 }
            );
        }

        mod pc8x8 {
            use super::*;

            $crate::testgen_matmul_stage_k!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_shape,
                PartitionsPerStage { m: 8, n: 8 }
            );
        }
    };
}
