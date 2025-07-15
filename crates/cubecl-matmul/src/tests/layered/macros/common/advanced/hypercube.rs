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
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::RowMajor))
                        .cube_count_plan(CubeCountPlanSelection::FromProblem)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlecol_fp {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleColMajor(2)))
                        .cube_count_plan(CubeCountPlanSelection::FromProblem)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod col_fl {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::ColMajor))
                        .cube_count_plan(CubeCountPlanSelection::Flattened)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlerow_fl {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleRowMajor(2)))
                        .cube_count_plan(CubeCountPlanSelection::Flattened)
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod row_sm_exact {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
                SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::RowMajor))
                        .cube_count_plan(CubeCountPlanSelection::Sm {
                            num_sms: 4,
                            sm_usage: SmAllocation::Exact,
                            cubes_first: false
                        })
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlecol_sm_exact {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
                SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleColMajor(2)))
                        .cube_count_plan(CubeCountPlanSelection::Sm {
                            num_sms: 4,
                            sm_usage: SmAllocation::Exact,
                            cubes_first: false
                        })
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod row_sm_full {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
                SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::RowMajor))
                        .cube_count_plan(CubeCountPlanSelection::Sm {
                            num_sms: 4,
                            sm_usage: SmAllocation::Full,
                            cubes_first: false
                        })
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlerow_cube_full {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
                SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleRowMajor(2)))
                        .cube_count_plan(CubeCountPlanSelection::Sm {
                            num_sms: 4,
                            sm_usage: SmAllocation::Full,
                            cubes_first: true
                        })
                        .build()
                )
            );
        }

        #[cfg(feature = "matmul_tests_hypercube")]
        mod swizzlerow_spread {
            use super::*;
            use $crate::components::batch::{
                CubeCountPlanSelection, GlobalOrder, GlobalOrderSelection, HypercubeSelection,
                SmAllocation,
            };

            $crate::testgen_matmul_partition_buffering!(
                $kind,
                $algorithm,
                $precision,
                $selection_builder.hypercube_config(
                    HypercubeSelection::builder(&$selection_builder.tiling_scheme.unwrap())
                        .global_order(GlobalOrderSelection::Fixed(GlobalOrder::SwizzleRowMajor(2)))
                        .cube_count_plan(CubeCountPlanSelection::Spread)
                        .build()
                )
            );
        }
    };
}
