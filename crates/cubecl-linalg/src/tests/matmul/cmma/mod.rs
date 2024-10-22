#![allow(missing_docs)]

pub mod matmul_internal;

#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_internal!(
            CmmaInstruction16_16_16,
            CmmaInstruction32_8_16,
            CmmaInstruction8_32_16,
            f32,
            half::f16,
            f32,
            32
        );
    };
}

#[macro_export]
macro_rules! testgen_cmma_mock {
    () => {
        use super::*;

        cubecl_linalg::testgen_cmma_internal!(
            DummyUnitInstruction16_16_16,
            DummyUnitInstruction32_8_16,
            DummyUnitInstruction8_32_16,
            f32,
            f32,
            f32,
            16
        );
    };
}
