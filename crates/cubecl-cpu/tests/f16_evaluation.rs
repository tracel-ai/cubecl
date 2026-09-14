//! The default, which is what a kernel gets with nothing set.

use std::sync::Once;

mod common;

static MODE: Once = Once::new();

/// Cleared rather than assumed absent, so that running the suite under a mode does not turn
/// these into failures that look like defects.
fn use_the_default() {
    MODE.call_once(|| unsafe { std::env::remove_var("CUBECL_CPU_F16_EVAL") });
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn host_has_f16_arithmetic() -> bool {
    std::arch::is_x86_feature_detected!("avx512fp16")
}

#[cfg(target_arch = "aarch64")]
fn host_has_f16_arithmetic() -> bool {
    std::arch::is_aarch64_feature_detected!("fp16")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn host_has_f16_arithmetic() -> bool {
    false
}

/// A host with f16 arithmetic of its own rounds every result, and any other holds the chain.
#[test]
fn the_default_holds_a_chain_only_without_host_f16_arithmetic() {
    use_the_default();
    let result = common::product_over_300();
    match host_has_f16_arithmetic() {
        true => assert!(result.is_infinite()),
        false => assert_eq!(result, 300.0),
    }
}

/// Neither default holds an accumulator, so a loop stops adding at the f16 step size.
#[test]
fn an_accumulator_is_not_held_by_default() {
    use_the_default();
    assert_eq!(common::accumulated(2048.0, 1.0, 1000), 2048.0);
}

/// A kernel with no f16 in it still compiles.
///
/// A barrier is a type with no element at all, and asking one for its scalar type is a panic in
/// the compiler rather than a wrong answer, so nothing downstream reports it.
#[test]
fn a_kernel_without_f16_is_untouched() {
    use_the_default();
    assert_eq!(common::barrier_reaches_the_store(), 1.0);
}
