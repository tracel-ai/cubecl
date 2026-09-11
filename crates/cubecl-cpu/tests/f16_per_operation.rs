//! `CUBECL_CPU_F16_EVAL=per-operation`, which rounds every result.

use std::sync::Once;

mod common;

static MODE: Once = Once::new();

fn set_mode() {
    MODE.call_once(|| unsafe { std::env::set_var("CUBECL_CPU_F16_EVAL", "per-operation") });
}

/// The intermediate is rounded, so `300 * 300` becomes an infinity and takes the quotient with
/// it.
///
/// This is the mode someone reaches for when a numerical difference has to be attributed, so it
/// has to actually differ from the default rather than quietly agree with it.
#[test]
fn an_intermediate_above_the_f16_maximum_becomes_infinite() {
    set_mode();
    assert!(common::product_over_300().is_infinite());
}
