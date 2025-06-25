#[macro_export]
macro_rules! testgen_matmul_hypercube {
    ($kind: ident, $algorithm: ty, $precision: ty, $selection_builder: expr) => {
        #[cfg(not(feature = "matmul_tests_hypercube"))]
        $crate::testgen_matmul_partition_buffering!(
            $kind,
            $algorithm,
            $precision,
            $selection_builder
        );

        #[cfg(feature = "matmul_tests_hypercube")]
        mod row_fp {
            use super::*;
            use $crate::components::batch::{CubeCountPlanConfig, GlobalOrder, HypercubeConfig};

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::RowMajor)
                        .cube_count_plan(CubeCountPlanConfig::FromProblem)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlecol_fp {
            use super::*;
            use $crate::components::batch::{CubeCountPlanConfig, GlobalOrder, HypercubeConfig};

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::SwizzleColMajor(2))
                        .cube_count_plan(CubeCountPlanConfig::FromProblem)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod col_fl {
            use super::*;
            use $crate::components::batch::{CubeCountPlanConfig, GlobalOrder, HypercubeConfig};

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::ColMajor)
                        .cube_count_plan(CubeCountPlanConfig::Flattened)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlerow_fl {
            use super::*;
            use $crate::components::batch::{CubeCountPlanConfig, GlobalOrder, HypercubeConfig};

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::SwizzleRowMajor(2))
                        .cube_count_plan(CubeCountPlanConfig::Flattened)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod row_sm_exact {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanConfig, GlobalOrder, HypercubeConfig, SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::RowMajor)
                        .cube_count_plan(CubeCountPlanConfig::SmFirst {
                            num_sms: 4,
                            sm_usage: SmAllocation::Exact
                        })
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlecol_sm_exact {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanConfig, GlobalOrder, HypercubeConfig, SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::SwizzleColMajor(2))
                        .cube_count_plan(CubeCountPlanConfig::SmFirst {
                            num_sms: 4,
                            sm_usage: SmAllocation::Exact
                        })
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod row_sm_full {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanConfig, GlobalOrder, HypercubeConfig, SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::RowMajor)
                        .cube_count_plan(CubeCountPlanConfig::SmFirst {
                            num_sms: 4,
                            sm_usage: SmAllocation::Full
                        })
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlerow_cube_full {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanConfig, GlobalOrder, HypercubeConfig, SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeConfig::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrder::SwizzleRowMajor(2))
                        .cube_count_plan(CubeCountPlanConfig::CubeFirst {
                            num_sms: 4,
                            sm_usage: SmAllocation::Full
                        })
                        .build()
                )
            );
        }
    };
}
