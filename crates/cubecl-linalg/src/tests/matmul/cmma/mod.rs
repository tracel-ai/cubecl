#![allow(missing_docs)]

pub mod matmul;
pub mod matmul_internal;
pub mod matmul_internal_mock;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_internal!();
        cubecl_linalg::testgen_cmma_matmul!();
    };
}

#[macro_export]
macro_rules! testgen_cmma_mock {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_internal_mock!();
    };
}
