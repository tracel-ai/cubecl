pub mod assign;
pub mod atomic;
pub mod barrier;
pub mod binary;
pub mod branch;
pub mod cluster;
pub mod cmma;
pub mod const_match;
pub mod constants;
pub mod debug;
pub mod different_rank;
pub mod enums;
pub mod index;
pub mod launch;
pub mod line;
pub mod metadata;
pub mod plane;
pub mod sequence;
pub mod slice;
pub mod sync_plane;
pub mod tensor;
pub mod tensormap;
pub mod topology;
pub mod traits;
pub mod unary;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        use $crate::Runtime;

        type FloatType = f32;
        type IntType = i32;
        type UintType = u32;

        $crate::testgen_float!();
        $crate::testgen_int!();
        $crate::testgen_untyped!();
    };
    ($f_def:ident: [$($float:ident),*], $i_def:ident: [$($int:ident),*], $u_def:ident: [$($uint:ident),*]) => {
        use $crate::Runtime;

        ::paste::paste! {
            $(mod [<$float _ty>] {
                use super::*;

                type FloatType = $float;
                type IntType = $i_def;
                type UintType = $u_def;

                $crate::testgen_float!();
            })*
            $(mod [<$int _ty>] {
                use super::*;

                type FloatType = $f_def;
                type IntType = $int;
                type UintType = $u_def;

                $crate::testgen_int!();
            })*
            $(mod [<$uint _ty>] {
                use super::*;

                type FloatType = $f_def;
                type IntType = $i_def;
                type UintType = $uint;

                $crate::testgen_uint!();
            })*
        }
        $crate::testgen_untyped!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_float {
    () => {
        cubecl_core::testgen_assign!();
        cubecl_core::testgen_barrier!();
        cubecl_core::testgen_binary!();
        cubecl_core::testgen_branch!();
        cubecl_core::testgen_const_match!();
        cubecl_core::testgen_different_rank!();
        cubecl_core::testgen_index!();
        cubecl_core::testgen_launch!();
        cubecl_core::testgen_line!();
        cubecl_core::testgen_plane!();
        cubecl_core::testgen_sequence!();
        cubecl_core::testgen_slice!();
        cubecl_core::testgen_unary!();
        cubecl_core::testgen_atomic_float!();
        cubecl_core::testgen_tensormap!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_int {
    () => {
        cubecl_core::testgen_unary_int!();
        cubecl_core::testgen_atomic_int!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_uint {
    () => {
        cubecl_core::testgen_const_match!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_untyped {
    () => {
        cubecl_core::testgen_cmma!();
        cubecl_core::testgen_metadata!();
        cubecl_core::testgen_topology!();

        cubecl_core::testgen_constants!();
        cubecl_core::testgen_sync_plane!();
        cubecl_core::testgen_tensor_indexing!();
        cubecl_core::testgen_debug!();
        cubecl_core::testgen_binary_untyped!();
        cubecl_core::testgen_cluster!();

        cubecl_core::testgen_enums!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_bytes {
    ($ty:ident: $($elem:expr),*) => {
        $ty::as_bytes(&[$($ty::new($elem),)*])
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_type {
    ($ty:ident: $($elem:expr),*) => {
        &[$($ty::new($elem),)*]
    };
}
