#[macro_export]
macro_rules! testgen_matmul_unit_specialized {
    ($algorithm: ty, $precision: ty, $tile: expr, $partition_size: expr, $stage_size: expr) => {
        use $crate::components::{LoadSpecializationConfig, SpecializationTensorConfig};

        mod mm {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Unit,
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

        #[cfg(feature = "matmul_tests_specialized")]
        mod ml {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Unit,
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

        #[cfg(feature = "matmul_tests_specialized")]
        mod lm {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Unit,
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

        #[cfg(feature = "matmul_tests_specialized")]
        mod ll {
            use super::*;

            $crate::testgen_matmul_layouts!(
                Unit,
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
