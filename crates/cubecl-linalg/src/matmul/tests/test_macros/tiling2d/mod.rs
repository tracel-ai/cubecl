#![allow(missing_docs)]

pub mod matmul;

#[macro_export]
macro_rules! testgen_tiling2d {
    () => {
        use super::*;

        cubecl_linalg::testgen_tiling2d_matmul!();
    };
}
