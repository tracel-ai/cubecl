#![allow(missing_docs)]

pub mod matmul_internal;
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

            pub type FloatT = $float;

            cubecl_linalg::testgen_matmul_internal!(
                Accelerated16x16x16,
                Accelerated32x8x16,
                Accelerated8x32x16,
                Config,
                FloatT,
                half::f16,
                f32,
                32
            );

            cubecl_linalg::testgen_matmul_launch!();
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

            pub type FloatT = $float;

            cubecl_linalg::testgen_matmul_internal!(
                PlaneMma16x16x16,
                PlaneMma32x8x16,
                PlaneMma8x32x16,
                Config,
                FloatT,
                f32,
                f32,
                32
            );

            cubecl_linalg::testgen_matmul_launch!();
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
