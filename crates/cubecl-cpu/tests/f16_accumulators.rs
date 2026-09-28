//! `f16_evaluation = "accumulators"`, which also holds a private f16 variable in f32.

use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::arithmetic_chains as chains;
use cubecl_cpu::CpuRuntime;
use cubecl_server::config::compilation::F16Evaluation;
use half::f16;

mod common;

fn client() -> Client {
    common::client_evaluating(Some(F16Evaluation::Accumulators))
}

/// The total is held in f32 across the back edge, not just within one iteration.
///
/// 2048 is where the f16 step size passes 1, so an f16 accumulator stops moving there and every
/// later addition is lost. It is the shape every f16 reduction has, and the one a mode confined
/// to straight-line code misses.
#[test]
fn an_accumulator_keeps_adding_past_the_f16_step_size() {
    let total = chains::accumulated::<f16>(&client(), 2048.0, 1.0, 1000);
    assert_eq!(total.to_f32(), 3048.0);
}

/// A second local on the path changes nothing about what the loop computes, so it must not
/// change whether the total is held. `mem2reg` collapses the copy, and it runs after the pass,
/// so the promotion has to see the two variables as one.
#[test]
fn a_copy_on_the_path_still_holds_the_accumulator() {
    let total = chains::accumulated_through_a_copy::<f16>(&client(), 2048.0, 1.0, 1000);
    assert_eq!(total.to_f32(), 3048.0);
}

/// Every lane of a vector accumulator is held, not only a scalar one.
#[test]
fn a_vector_accumulator_keeps_adding_past_the_f16_step_size() {
    let lanes = chains::accumulated_vector::<f16>(&client(), 2048.0, 1.0, 1000);
    assert_eq!(lanes.map(f16::to_f32), [3048.0; chains::LANES]);
}

/// A `let mut` outside any loop is held too: its store and its load are both converts it removes.
#[test]
fn a_let_mut_is_held() {
    let result = chains::product_over_through_a_let_mut::<f16>(&client(), [300.0; 3]);
    assert_eq!(result.to_f32(), 300.0);
}

/// A store under an `if` is paid once, so it cannot outvote the converts at the boundary the way a
/// store paid every iteration would. Held, the variable would carry `300 * 300` into the division.
#[test]
fn a_branch_is_not_a_loop() {
    let result = chains::product_on_a_branch::<f16>(&client(), [300.0; 3]);
    assert!(result.is_infinite());
}

/// A barrier has no element type, and the variable check still has to pass over it.
#[test]
fn a_kernel_without_f16_is_untouched() {
    cubecl_core::runtime_tests::barrier::test_async_memcpy::<CpuRuntime, f32>(client());
}
