use cubecl_common::quant::scheme::{QuantParam, QuantScheme};
use cubecl_core::prelude::Scalar;

/// Run an arbitrary function with the quantization types from the scheme.
/// Useful when concrete types aren't available.
pub trait RunWithQuantType {
    type Output;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output;
}

/// Panic when the scale bindings and the scheme's levels disagree.
///
/// Every level binds a scale buffer of its own, so nothing ties the bindings to the scheme: a
/// missing level is dropped from the reconstruction and every value comes back short by that
/// factor, an extra one is a caller quantizing differently than the scheme it passed.
///
/// Every outer level is further constrained by what this reader serves: its block must be full,
/// since only the innermost level's scales are addressed per position, and it binds as f32 rather
/// than being read as f32 bytes. An outer level has one scale for the whole tensor, so a narrower
/// type saves nothing and only reintroduces rounding error.
pub fn check_scale_bindings(scheme: &QuantScheme, bindings: usize) {
    let levels = scheme.levels();
    assert!(
        bindings == levels.len(),
        "a scheme with {} scale level(s) takes as many scale bindings, but {bindings} were provided",
        levels.len(),
    );
    check_outer_levels(scheme);
}

/// The outer-level half of [`check_scale_bindings`], for a consumer holding outer scales already
/// folded into a register rather than as countable bindings.
pub fn check_outer_levels(scheme: &QuantScheme) {
    for outer in &scheme.levels()[1..] {
        assert!(
            outer.block.is_full(),
            "the quantized view only serves outer levels covering the whole tensor, not {:?}",
            outer.block,
        );
        assert!(
            outer.param == QuantParam::F32,
            "an outer scale binds as f32, but the scheme stores it as {:?}",
            outer.param,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::check_scale_bindings;
    use cubecl_common::quant::scheme::{QuantParam, QuantScheme, ScaleLevels};

    fn scheme(levels: ScaleLevels) -> QuantScheme {
        QuantScheme::default().with_scales(levels)
    }

    #[test]
    fn a_one_level_scheme_takes_one_binding() {
        check_scale_bindings(&scheme(ScaleLevels::tensor(QuantParam::F32)), 1);
        check_scale_bindings(&scheme(ScaleLevels::block([32], QuantParam::F32)), 1);
    }

    #[test]
    fn a_two_level_scheme_takes_two_bindings() {
        check_scale_bindings(
            &scheme(ScaleLevels::block([32], QuantParam::F32).and_tensor(QuantParam::F32)),
            2,
        );
    }

    /// The binding is f32, so a level naming another param would have its scale read as f32 bytes.
    #[test]
    #[should_panic(expected = "binds as f32, but")]
    fn a_two_level_scheme_storing_the_tensor_scale_narrower_is_rejected() {
        check_scale_bindings(
            &scheme(ScaleLevels::block([32], QuantParam::F32).and_tensor(QuantParam::BF16)),
            2,
        );
    }

    #[test]
    #[should_panic(expected = "takes as many scale bindings, but 1 were provided")]
    fn a_two_level_scheme_with_one_binding_is_rejected() {
        // Would otherwise dequantize against the block scales alone, dropping the per-tensor factor.
        check_scale_bindings(
            &scheme(ScaleLevels::block([32], QuantParam::F32).and_tensor(QuantParam::F32)),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "takes as many scale bindings, but 2 were provided")]
    fn a_one_level_scheme_with_two_bindings_is_rejected() {
        check_scale_bindings(&scheme(ScaleLevels::tensor(QuantParam::F32)), 2);
    }
}
