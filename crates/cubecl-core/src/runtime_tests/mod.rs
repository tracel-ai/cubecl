pub mod assign;
pub mod binary;
pub mod branch;
pub mod cmma;
pub mod const_match;
pub mod constants;
pub mod different_rank;
pub mod launch;
pub mod metadata;
pub mod sequence;
pub mod slice;
pub mod subcube;
pub mod topology;
pub mod unary;

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
        cubecl_core::testgen_branch!();
        cubecl_core::testgen_constants!();
        cubecl_core::testgen_topology!();
        cubecl_core::testgen_metadata!();
        cubecl_core::testgen_sequence!();
        cubecl_core::testgen_unary!();
        cubecl_core::testgen_binary!();
        cubecl_core::testgen_different_rank!();
        cubecl_core::testgen_const_match!();
    };
}
