use cubecl_common::quant::scheme::{QuantLevel, QuantParam};
use cubecl_core::prelude::Scalar;

/// Run an arbitrary function with the quantization types from the scheme.
/// Useful when concrete types aren't available.
pub trait RunWithQuantType {
    type Output;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output;
}

/// Panic when the per-tensor scale binding and the level disagree.
///
/// The per-tensor scale binds as a buffer of its own, so nothing ties it to the level: a missing
/// one is dropped from the reconstruction and every value comes back short by that factor, an
/// extra one is a caller quantizing differently than the scheme it passed.
///
/// The binding is f32, and a level storing the scale in anything else is rejected rather than read
/// as f32 bytes. There is one per-tensor scale for a whole tensor, so a narrower type saves nothing
/// and only reintroduces rounding error.
pub fn check_global_bindings(level: QuantLevel, global_provided: bool) {
    match (level.global_param(), global_provided) {
        (None, false) | (Some(QuantParam::F32), true) => {}
        (Some(_), false) => {
            panic!("{level:?} takes a per-tensor scale, but no global was provided")
        }
        (None, true) => {
            panic!("global was provided, but {level:?} does not take a per-tensor scale")
        }
        (Some(param), true) => {
            panic!("the per-tensor scale binds as f32, but {level:?} stores it as {param:?}")
        }
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

    #[test]
    fn a_two_level_scheme_takes_an_f32_global() {
        check_global_bindings(QuantLevel::block_tensor([32], QuantParam::F32), true);
    }

    /// The binding is f32, so a level naming another param would have its scale read as f32 bytes.
    #[test]
    #[should_panic(expected = "binds as f32, but")]
    fn a_two_level_scheme_storing_the_global_narrower_is_rejected() {
        check_global_bindings(QuantLevel::block_tensor([32], QuantParam::BF16), true);
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
