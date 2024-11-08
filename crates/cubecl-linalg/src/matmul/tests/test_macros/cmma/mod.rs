#![allow(missing_docs)]

pub mod matmul_algorithm;
pub mod matmul_launch;

#[macro_export]
macro_rules! testgen_cmma_matmul {
    () => {
        mod cmma_matmul {
            $crate::testgen_cmma_matmul!(f32);
        }
    };

    ($float:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;
            use cubecl_linalg::matmul::components::tile::accelerated::*;
            use cubecl_core::prelude::*;
            use cubecl_linalg::matmul::{
                components::{
                    batch, global,
                    stage::{self, *},
                    tile::plane::PlaneMma16x16x16,
                    MatmulProblem, MatrixLayout,
                },
                kernels::matmul::{self, AdvancedConfig},
                tests::cmma_matmul::matmul_test_launcher::test_matmul_internal,
            };
            use cubecl_core::prelude::*;

            pub type FloatT = $float;

            cubecl_linalg::matmul_test_define!(
                Accelerated16x16x16,
                Accelerated32x8x16,
                Accelerated8x32x16,
                FloatT,
                half::f16,
                f32,
                32
            );

            cubecl_linalg::testgen_matmul_launch!(
                FloatT,
            );
    };

    ([$($float:ident),*]) => {
        mod cmma_matmul {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_cmma_matmul!($float);
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_plane_mma {
    () => {
        mod plane_mma_matmul {
            $crate::testgen_plane_mma!(f32);
        }
    };

    ($float:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;
            use cubecl_linalg::matmul::components::tile::plane::*;
            use cubecl_linalg::matmul::{
                components::{
                    batch, global,
                    stage::{self, *},
                    tile::plane::PlaneMma16x16x16,
                    MatmulProblem, MatrixLayout,
                },
                kernels::matmul::{self, AdvancedConfig},
                tests::cmma_matmul::matmul_test_launcher::test_matmul_internal,
            };
            use cubecl_core::prelude::*;

            pub type FloatT = $float;

            cubecl_linalg::matmul_test_define!(
                PlaneMma16x16x16,
                PlaneMma32x8x16,
                PlaneMma8x32x16,
                FloatT,
                f32,
                f32,
                32
            );

            cubecl_linalg::testgen_matmul_launch!(
               FloatT,
            );
    };

    ([$($float:ident),*]) => {
        mod plane_mma_matmul {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_plane_mma!($float);
                })*
            }
        }
    };
}
