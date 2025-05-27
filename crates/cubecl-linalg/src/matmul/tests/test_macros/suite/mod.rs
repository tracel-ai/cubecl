//! Generates tests by combining parameters in the following order:
//!
//! - `kind`: high-level matmul category (e.g., `accelerated`, `tma`, `unit`, `quantized`)
//! - `algorithm`: compute/loading strategy (e.g., `double_buffering_tilewise`, `simply_cyclic`)
//! - `precision`: data type (e.g., `f16`, `f32`)
//! - `tile`: instruction tile dimensions in M/N/K
//! - `partition`: accumulator count per execution primitive (number of tiles in M/N)
//! - `stage`: shared memory shape (number of tiles in M/N/K)
//! - `layouts`: operand layouts (row-major (r) or column-major (c) for lhs and rhs)
//! - `problem`: actual matrix dimensions M/N/K

pub mod common;
pub mod plane_accelerated;
pub mod quantized;
pub mod tma;
pub mod unit;

#[macro_export]
macro_rules! testgen_matmul_plane_accelerated {
    () => {
        mod matmul_plane_accelerated {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            $crate::testgen_matmul_plane_accelerated_algorithm!();
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_unit {
    () => {
        mod matmul_unit {
            use super::*;

            $crate::testgen_matmul_unit_algorithm!();
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_tma {
    () => {
        mod matmul_tma {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            $crate::testgen_matmul_tma_algorithm!();
        }
    };
}

#[macro_export]
macro_rules! testgen_matmul_quantized {
    () => {
        mod matmul_quantized {
            use super::*;
            type TMM = $crate::matmul::components::tile::accelerated_matmul::AcceleratedMatmul;

            $crate::testgen_matmul_quantized_algorithm!();
        }
    };
}
