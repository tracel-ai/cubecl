#![allow(missing_docs)]

pub mod matmul;
pub mod matmul_internal;

#[macro_export]
macro_rules! testgen_all {
    () => {
        mod lac {
            use super::*;

            cubecl_lac::testgen_matmul!();
            cubecl_lac::testgen_matmul_internal!();
        }
    };
}
