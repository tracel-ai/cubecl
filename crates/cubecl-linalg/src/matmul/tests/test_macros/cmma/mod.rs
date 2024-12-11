#![allow(missing_docs)]

pub mod matmul_algorithm;
pub mod matmul_launch;

#[macro_export]
macro_rules! testgen_cmma_matmul {
    () => {
        #[allow(non_snake_case)]
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
                    SingleMatmulSpec,
                    MatmulSpec,
                    batch, global,
                    stage::{self, *},
                    tile::plane::PlaneMma16x16x16,
                    MatmulProblem, MatrixLayout,
                },
                kernels::matmul::{self, AdvancedConfig},
                tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm,
            };
            use cubecl_core::prelude::*;

            pub type FloatT = $float;
            pub type Spec = SingleMatmulSpec<32, FloatT, half::f16, f32>;
            pub type EG = FloatT;
            pub type ES = half::f16;
            pub type EA = f32;

            cubecl_linalg::matmul_test_define!(
                Accelerated16x16x16,
                Accelerated32x8x16,
                Accelerated8x32x16
            );

            cubecl_linalg::testgen_matmul_launch!(
                FloatT
            );
    };

    ([$($float:ident),*]) => {
        #[allow(non_snake_case)]
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
        #[allow(non_snake_case)]
        mod plane_mma_matmul {
            $crate::testgen_plane_mma!(f32, f32);
        }
    };

    ($float:ident, $float_stage:ident) => {
            use super::*;
            use cubecl_linalg::matmul::tests;
            use cubecl_linalg::matmul::components::tile::plane::*;
            use cubecl_linalg::matmul::{
                components::{
                    SingleMatmulSpec,
                    MatmulSpec,
                    batch, global,
                    stage::{self, *},
                    tile::plane::PlaneMma16x16x16,
                    MatmulProblem, MatrixLayout,
                },
                kernels::matmul::{self, AdvancedConfig},
                tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm,
            };
            use cubecl_core::prelude::*;

            pub type FloatGlobal = $float;
            pub type FloatStage = $float_stage;
            pub type Spec = SingleMatmulSpec<32, FloatGlobal, FloatStage, f32>;
            pub type EG = FloatGlobal;
            pub type ES = FloatStage;
            pub type EA = f32;

            cubecl_linalg::matmul_test_define!(
                PlaneMma16x16x16,
                PlaneMma32x8x16,
                PlaneMma8x32x16
            );

            cubecl_linalg::testgen_matmul_launch!(
                FloatGlobal
            );
    };

    ([$($float:ident),*], $float_stage:ident) => {
        ::paste::paste! {
            $(
                // Generate a unique module for each `float` type with the single `float_stage`
                #[allow(non_snake_case)]
                mod [<plane_mma_matmul_ $float _ $float_stage>] {
                    use super::*;
                    $crate::testgen_plane_mma!($float, $float_stage);
                }
            )*
        }
    };
}
