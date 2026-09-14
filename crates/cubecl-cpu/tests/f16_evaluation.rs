//! The default, which is what a kernel gets with nothing set.

use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::arithmetic_chains as chains;
use half::f16;

mod common;

fn client() -> Client {
    common::client_evaluating(None)
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
    let result = chains::product_over::<f16>(&client(), [300.0; 3]);
    match host_has_f16_arithmetic() {
        true => assert!(result.is_infinite()),
        false => assert_eq!(result.to_f32(), 300.0),
    }
}

/// Neither default holds an accumulator, so a loop stops adding at the f16 step size.
#[test]
fn an_accumulator_is_not_held_by_default() {
    let total = chains::accumulated::<f16>(&client(), 2048.0, 1.0, 1000);
    assert_eq!(total.to_f32(), 2048.0);
}
