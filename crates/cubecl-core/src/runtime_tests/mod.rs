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
        use $crate::Runtime;

        type FloatT = f32;
        type IntT = i32;
        type UintT = u32;

        cubecl_core::testgen_assign!();
        cubecl_core::testgen_branch!();
        cubecl_core::testgen_const_match!();
        cubecl_core::testgen_different_rank!();
        cubecl_core::testgen_launch!();

        $crate::testgen_untyped!();
    };
    ($f_def:ident: [$($float:ident),*], $i_def:ident: [$($int:ident),*], $u_def:ident: [$($uint:ident),*]) => {
        use $crate::Runtime;

        ::paste::paste! {
            $(mod [<$float _ty>] {
                type FloatT = $float;
                type IntT = $i_def;
                type UintT = $u_def;

                $crate::testgen_float!();
            })*
            $(mod [<$int _ty>] {
                type FloatT = $f_def;
                type IntT = $int;
                type UintT = $u_def;

                $crate::testgen_int!();
            })*
            $(mod [<$uint _ty>] {
                type FloatT = $f_def;
                type IntT = $i_def;
                type UintT = $uint;

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
        cubecl_core::testgen_binary!();
        cubecl_core::testgen_branch!();
        cubecl_core::testgen_const_match!();
        cubecl_core::testgen_different_rank!();
        cubecl_core::testgen_launch!();
        cubecl_core::testgen_sequence!();
        cubecl_core::testgen_slice!();
        cubecl_core::testgen_subcube!();
        cubecl_core::testgen_unary!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_int {
    () => {};
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
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_bytes {
    ($ty:ident: $($elem:expr),*) => {
        F::as_bytes(&[$($ty::new($elem),)*])
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_type {
    ($ty:ident: $($elem:expr),*) => {
        &[$($ty::new($elem),)*]
    };
}
