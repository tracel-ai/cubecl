#[macro_export]
macro_rules! testgen_matmul_accelerated_specialized {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr, $stage_size: expr) => {
        use $crate::components::{LoadSpecializationConfig, SpecializationTensorConfig};

        mod mm {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::MainFlowOnly,
                    rhs: SpecializationTensorConfig::MainFlowOnly,
                }
            );
        }

        mod ml {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::MainFlowOnly,
                    rhs: SpecializationTensorConfig::LoadFlowOnly,
                }
            );
        }

        mod lm {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::LoadFlowOnly,
                    rhs: SpecializationTensorConfig::MainFlowOnly,
                }
            );
        }

        mod ll {
            use super::*;

            $crate::testgen_matmul_layouts!(
                PlaneAccelerated,
                $algorithm,
                $precision,
                $tile,
                $partition_size,
                $stage_size,
                LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::LoadFlowOnly,
                    rhs: SpecializationTensorConfig::LoadFlowOnly,
                }
            );
        }
    };
}
