//! `f16_evaluation = "chain"`, which holds a chain of arithmetic in f32 and rounds at its end.

use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::arithmetic_chains as chains;
use cubecl_server::config::compilation::F16Evaluation;
use half::f16;

mod common;

fn client() -> Client {
    common::client_evaluating(Some(F16Evaluation::Chain))
}

/// An f16 intermediate above the f16 maximum survives, which changes range rather than precision.
///
/// gcc and clang hold a `_Float16` expression the same way, and round after every operation only
/// under `-fexcess-precision=16`.
#[test]
fn a_chain_is_held_in_f32() {
    let result = chains::product_over::<f16>(&client(), [300.0; 3]);
    assert_eq!(result.to_f32(), 300.0);
}

/// An immutable `let` is an SSA value, so the chain runs through it, where C would round at the
/// assignment.
#[test]
fn a_let_does_not_end_a_chain() {
    let result = chains::product_over_through_a_let::<f16>(&client(), [300.0; 3]);
    assert_eq!(result.to_f32(), 300.0);
}

/// A `let mut` is a variable, and a store rounds.
#[test]
fn a_let_mut_ends_a_chain() {
    let result = chains::product_over_through_a_let_mut::<f16>(&client(), [300.0; 3]);
    assert!(result.is_infinite());
}

/// A loop-carried accumulator is not held, which is where the mode stops.
///
/// 2048 is where the f16 step size passes 1, so an f16 accumulator stops moving there. No C
/// compiler carries excess precision across an assignment either, and holding it costs vector
/// registers, so it is asked for separately.
#[test]
fn an_accumulator_is_not_held() {
    let total = chains::accumulated::<f16>(&client(), 2048.0, 1.0, 1000);
    assert_eq!(total.to_f32(), 2048.0);
}

/// Nor is one that passes through a second local, which is the same loop written differently.
#[test]
fn a_copy_on_the_path_is_not_held() {
    let total = chains::accumulated_through_a_copy::<f16>(&client(), 2048.0, 1.0, 1000);
    assert_eq!(total.to_f32(), 2048.0);
}
