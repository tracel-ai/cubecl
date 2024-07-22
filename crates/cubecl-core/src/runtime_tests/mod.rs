pub mod assign;
pub mod cmma;
pub mod launch;
pub mod slice;
pub mod subcube;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        use cubecl_core::prelude::*;

        cubecl_core::testgen_subcube!();
        cubecl_core::testgen_launch!();
        cubecl_core::testgen_cmma!();
        cubecl_core::testgen_slice!();
        cubecl_core::testgen_assign!();
    };
}
