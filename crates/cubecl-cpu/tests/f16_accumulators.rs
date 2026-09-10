//! `CUBECL_CPU_F16_EVAL=accumulators`, which extends the default across a loop.

use std::sync::Once;

mod common;

static MODE: Once = Once::new();

fn set_mode() {
    MODE.call_once(|| unsafe { std::env::set_var("CUBECL_CPU_F16_EVAL", "accumulators") });
}

/// The total is held in f32 across the back edge, not just within one iteration.
///
/// 2048 is where the f16 step size passes 1, so an f16 accumulator stops moving there and every
/// later addition is lost. It is the shape every f16 reduction has, and the one a mode confined
/// to straight-line code misses.
#[test]
fn an_accumulator_keeps_adding_past_the_f16_step_size() {
    set_mode();
    assert_eq!(common::accumulated(2048.0, 1.0, 1000), 3048.0);
}

/// A second local on the path changes nothing about what the loop computes, so it must not
/// change whether the total is held. `mem2reg` collapses the copy, and it runs after the pass,
/// so the promotion has to see the two variables as one.
#[test]
fn a_copy_on_the_path_still_holds_the_accumulator() {
    set_mode();
    assert_eq!(
        common::accumulated_through_a_copy(2048.0, 1.0, 1000),
        3048.0
    );
}

/// A kernel with no f16 in it still compiles, with the variable check reached as well.
#[test]
fn a_kernel_without_f16_is_untouched() {
    set_mode();
    assert_eq!(common::barrier_reaches_the_store(), 1.0);
}
