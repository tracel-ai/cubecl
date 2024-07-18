#![allow(missing_docs)]

pub mod matmul;
pub mod matmul_internal;

#[macro_export]
macro_rules! testgen_all {
    () => {
        mod linalg {
            use super::*;

            cubecl_linalg::testgen_matmul!();
            cubecl_linalg::testgen_matmul_internal!();
        }
    };
}
