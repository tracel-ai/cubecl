#![allow(missing_docs)]

pub mod matmul;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_matmul!();
    };
}
