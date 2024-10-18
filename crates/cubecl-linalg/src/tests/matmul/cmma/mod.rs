#![allow(missing_docs)]

pub mod matmul_internal;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;
    };
}

#[macro_export]
macro_rules! testgen_cmma_mock {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_internal!();
    };
}
