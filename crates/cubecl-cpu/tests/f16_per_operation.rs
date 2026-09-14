//! `f16_evaluation = "per-operation"`, which rounds every result.

use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::arithmetic_chains as chains;
use cubecl_server::config::compilation::F16Evaluation;
use half::f16;

mod common;

fn client() -> Client {
    common::client_evaluating(Some(F16Evaluation::PerOperation))
}

/// The intermediate is rounded, so `300 * 300` becomes an infinity and takes the quotient with
/// it.
///
/// This is the mode someone reaches for when a numerical difference has to be attributed, so it
/// has to actually differ from `chain` rather than quietly agree with it.
#[test]
fn an_intermediate_above_the_f16_maximum_becomes_infinite() {
    let result = chains::product_over::<f16>(&client(), [300.0; 3]);
    assert!(result.is_infinite());
}
