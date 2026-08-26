//! The compiler toggle is read from the environment, so it must default safely
//! and must not silently pick LLVM when the feature is compiled out.

use cubecl_hip::HipCompiler;

#[test]
fn defaults_to_cpp_when_unset() {
    unsafe { std::env::remove_var("CUBECL_HIP_COMPILER") };
    assert!(matches!(HipCompiler::default(), HipCompiler::Cpp(_)));
}

#[test]
fn unknown_value_falls_back_to_cpp() {
    unsafe { std::env::set_var("CUBECL_HIP_COMPILER", "nonsense") };
    assert!(matches!(HipCompiler::default(), HipCompiler::Cpp(_)));
    unsafe { std::env::remove_var("CUBECL_HIP_COMPILER") };
}

#[cfg(feature = "llvm")]
#[test]
fn llvm_selects_the_pliron_compiler() {
    unsafe { std::env::set_var("CUBECL_HIP_COMPILER", "llvm") };
    assert!(matches!(HipCompiler::default(), HipCompiler::Llvm(_)));
    unsafe { std::env::remove_var("CUBECL_HIP_COMPILER") };
}
