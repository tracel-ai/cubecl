//! The default, which is what a kernel gets with nothing set.

use std::sync::Once;

mod common;

static MODE: Once = Once::new();

/// Cleared rather than assumed absent, so that running the suite under a mode does not turn
/// these into failures that look like defects.
fn use_the_default() {
    MODE.call_once(|| unsafe { std::env::remove_var("CUBECL_CPU_F16_EVAL") });
}

/// An f16 intermediate above the f16 maximum survives, so a chain is held in f32 unless asked
/// otherwise.
///
/// This is the part of the policy that changes range rather than precision. gcc and clang hold a
/// `_Float16` expression the same way, and round after every operation only under
/// `-fexcess-precision=16`.
#[test]
fn a_chain_is_held_in_f32_by_default() {
    use_the_default();
    assert_eq!(common::product_over_300(), 300.0);
}

/// An immutable `let` is an SSA value, so the chain runs through it, where C would round at the
/// assignment.
#[test]
fn a_let_does_not_end_a_chain() {
    use_the_default();
    assert_eq!(common::product_over_300_through_a_let(), 300.0);
}

/// A `let mut` is a variable, and a store rounds.
#[test]
fn a_let_mut_ends_a_chain() {
    use_the_default();
    assert!(common::product_over_300_through_a_let_mut().is_infinite());
}

/// A loop-carried accumulator is not held, which is where the default stops.
///
/// 2048 is where the f16 step size passes 1, so an f16 accumulator stops moving there. No C
/// compiler carries excess precision across an assignment either, and holding it costs vector
/// registers, so it is asked for separately.
#[test]
fn an_accumulator_is_not_held_by_default() {
    use_the_default();
    assert_eq!(common::accumulated(2048.0, 1.0, 1000), 2048.0);
}

/// Nor is one that passes through a second local, which is the same loop written differently.
#[test]
fn a_copy_on_the_path_is_not_held_by_default() {
    use_the_default();
    assert_eq!(
        common::accumulated_through_a_copy(2048.0, 1.0, 1000),
        2048.0
    );
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
