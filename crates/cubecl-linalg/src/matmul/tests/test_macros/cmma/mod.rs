#![allow(missing_docs)]

pub mod matmul_internal;
pub mod matmul_launch;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;
        use cubecl_linalg::matmul::components::tile::cmma::*;

        cubecl_linalg::testgen_matmul_internal!(
            CmmaInstruction16_16_16,
            CmmaInstruction32_8_16,
            CmmaInstruction8_32_16,
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
