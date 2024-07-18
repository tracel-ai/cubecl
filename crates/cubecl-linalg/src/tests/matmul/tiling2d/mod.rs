#![allow(missing_docs)]

pub mod matmul;
pub mod matmul_internal;

#[macro_export]
macro_rules! testgen_tiling2d {
    () => {
        use super::*;

        cubecl_linalg::testgen_tiling2d_matmul!();
        cubecl_linalg::testgen_tiling2d_internal!();
    };
}
