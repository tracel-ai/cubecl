#[macro_export]
macro_rules! testgen_matmul_specialized {
    ($kind: ident, $algorithm: ty, $precision: ty, $selection_builder: expr) => {
        use $crate::components::global::{LoadSpecializationConfig, SpecializationTensorConfig};

        #[cfg(not(feature = "matmul_tests_specialized"))]
        $crate::testgen_matmul_hypercube!(
            $kind,
            $algorithm,
            $precision,
            $selection_builder.load_specialization_config(LoadSpecializationConfig {
                lhs: SpecializationTensorConfig::MainFlowOnly,
                rhs: SpecializationTensorConfig::MainFlowOnly,
            })
        );

        #[cfg(feature = "matmul_tests_specialized")]
        mod mm {
            use super::*;

            $crate::testgen_matmul_hypercube!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.load_specialization_config(LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::MainFlowOnly,
                    rhs: SpecializationTensorConfig::MainFlowOnly,
                })
            );
        }

        #[cfg(feature = "matmul_tests_specialized")]
        mod ml {
            use super::*;

            $crate::testgen_matmul_hypercube!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.load_specialization_config(LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::MainFlowOnly,
                    rhs: SpecializationTensorConfig::LoadFlowOnly,
                })
            );
        }

        #[cfg(feature = "matmul_tests_specialized")]
        mod lm {
            use super::*;

            $crate::testgen_matmul_hypercube!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.load_specialization_config(LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::LoadFlowOnly,
                    rhs: SpecializationTensorConfig::MainFlowOnly,
                })
            );
        }

        #[cfg(feature = "matmul_tests_specialized")]
        mod ll {
            use super::*;

            $crate::testgen_matmul_hypercube!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.load_specialization_config(LoadSpecializationConfig {
                    lhs: SpecializationTensorConfig::LoadFlowOnly,
                    rhs: SpecializationTensorConfig::LoadFlowOnly,
                })
            );
        }
    };
}
