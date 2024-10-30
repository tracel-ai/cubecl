#![allow(missing_docs)]

pub mod matmul_internal;
pub mod matmul_launch;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;
        use cubecl_linalg::matmul::components::tile::accelerated::*;

        cubecl_linalg::testgen_matmul_internal!(
            Accelerated16x16x16,
            Accelerated32x8x16,
            Accelerated8x32x16,
            f32,
            half::f16,
            f32,
            32
        );

        cubecl_linalg::testgen_matmul_launch!();
    };
}

#[macro_export]
macro_rules! testgen_plane_mma {
    () => {
        use super::*;
        use cubecl_linalg::matmul::components::tile::plane::*;

        cubecl_linalg::testgen_matmul_internal!(
            PlaneMma16x16x16,
            PlaneMma32x8x16,
            PlaneMma8x32x16,
            f32,
            f32,
            f32,
            32
        );

        cubecl_linalg::testgen_matmul_launch!();
    };
}
