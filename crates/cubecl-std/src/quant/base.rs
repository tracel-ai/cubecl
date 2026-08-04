use cubecl_common::quant::scheme::{QuantLevel, QuantParam};
use cubecl_core::prelude::Scalar;

/// Run an arbitrary function with the quantization types from the scheme.
/// Useful when concrete types aren't available.
pub trait RunWithQuantType {
    type Output;

    /// Whether the caller bound a per-tensor scale, checked against the level by
    /// [`check_global_bindings`].
    fn global_provided(&self) -> bool;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output;
}

/// Panic when the per-tensor scale binding and the level disagree.
///
/// The per-tensor scale binds as its own view, so nothing ties it to the level: a missing one is
/// dropped from the reconstruction and every value comes back short by that factor, an extra one is
/// a caller quantizing differently than the scheme it passed. Levels are matched exhaustively so a
/// new one has to make a decision here rather than inherit silence.
pub fn check_global_bindings(level: QuantLevel, global_provided: bool) {
    match (level.global_param(), global_provided) {
        (Some(param), true) => assert_eq!(
            param,
            QuantParam::F32,
            "the per-tensor scale is read as f32, but {level:?} stores it as {param:?}"
        ),
        (Some(_), false) => {
            panic!("{level:?} takes a per-tensor scale, but no global was provided")
        }
        (None, true) => {
            panic!("global was provided, but {level:?} does not take a per-tensor scale")
        }
        (None, false) => {}
    }
}
