#![allow(missing_docs)]

pub mod cmma;
pub mod tiling2d;

#[macro_export]
macro_rules! testgen_all {
    () => {
        mod linalg {
            use super::*;
            use cubecl_linalg::matmul::tests;

            cubecl_linalg::testgen_cmma!();
            cubecl_linalg::testgen_tiling2d!();
        }
    };
}
