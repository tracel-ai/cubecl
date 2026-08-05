use cubecl_common::quant::scheme::QuantLevel;
use cubecl_core::prelude::Scalar;

/// Run an arbitrary function with the quantization types from the scheme.
/// Useful when concrete types aren't available.
pub trait RunWithQuantType {
    type Output;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output;
}

/// Panic when the per-tensor scale binding and the level disagree.
///
/// The per-tensor scale binds as its own view, so nothing ties it to the level: a missing one is
/// dropped from the reconstruction and every value comes back short by that factor, an extra one is
/// a caller quantizing differently than the scheme it passed. Levels are matched exhaustively so a
/// new one has to make a decision here rather than inherit silence.
pub fn check_global_bindings(level: QuantLevel, global_provided: bool) {
    let takes_global = match level {
        QuantLevel::Tensor | QuantLevel::Block(_) => false,
        QuantLevel::BlockTensor { .. } => true,
    };

    match (takes_global, global_provided) {
        (true, false) => panic!("{level:?} takes a per-tensor scale, but no global was provided"),
        (false, true) => {
            panic!("global was provided, but {level:?} does not take a per-tensor scale")
        }
        (true, true) | (false, false) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::check_global_bindings;
    use cubecl_common::quant::scheme::{QuantLevel, QuantParam};

    #[test]
    fn a_one_level_scheme_takes_no_global() {
        check_global_bindings(QuantLevel::Tensor, false);
        check_global_bindings(QuantLevel::block([32]), false);
    }

    /// The per-tensor scale is read through its own binding and cast to f32, so any param the
    /// level can hold is launchable.
    #[test]
    fn a_two_level_scheme_takes_a_global_of_any_param() {
        for param in [
            QuantParam::F32,
            QuantParam::F16,
            QuantParam::BF16,
            QuantParam::UE8M0,
            QuantParam::UE4M3,
        ] {
            check_global_bindings(QuantLevel::block_tensor([32], param), true);
        }
    }

    #[test]
    #[should_panic(expected = "takes a per-tensor scale, but no global was provided")]
    fn a_two_level_scheme_without_a_global_is_rejected() {
        // Would otherwise dequantize against the block scales alone, dropping the per-tensor factor.
        check_global_bindings(QuantLevel::block_tensor([32], QuantParam::F32), false);
    }

    #[test]
    #[should_panic(expected = "does not take a per-tensor scale")]
    fn a_one_level_scheme_with_a_global_is_rejected() {
        check_global_bindings(QuantLevel::Tensor, true);
    }
}
