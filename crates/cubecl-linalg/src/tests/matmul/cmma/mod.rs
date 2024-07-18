#![allow(missing_docs)]

pub mod matmul;
pub mod matmul_internal;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_matmul!();
        cubecl_linalg::testgen_cmma_internal!();
    };
}
