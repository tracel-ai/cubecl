use cubecl_common::quant::scheme::{QuantScheme, ScaleDtype};
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
/// The outer level is further constrained by what this reader serves: it binds as f32 rather than
/// being read as f32 bytes. It has one scale for the whole tensor, so a narrower type saves
/// nothing and only reintroduces rounding error.
pub fn check_scale_bindings(scheme: &QuantScheme, bindings: usize) {
    let levels = scheme.num_levels();
    assert!(
        bindings == levels,
        "a scheme with {levels} scale level(s) takes as many scale bindings, but {bindings} were provided",
    );
    check_outer_levels(scheme);
}

/// The outer-level half of [`check_scale_bindings`], for a consumer holding outer scales already
/// folded into a register rather than as countable bindings.
pub fn check_outer_levels(scheme: &QuantScheme) {
    if scheme.block_scale().is_some()
        && let Some(tensor) = scheme.tensor_scale()
    {
        assert!(
            tensor == ScaleDtype::F32,
            "an outer scale binds as f32, but the scheme stores it as {tensor:?}",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::check_scale_bindings;
    use cubecl_common::quant::scheme::{QuantScheme, ScaleDtype};

    #[test]
    fn a_one_level_scheme_takes_one_binding() {
        check_scale_bindings(&QuantScheme::default().per_tensor(ScaleDtype::F32), 1);
        check_scale_bindings(&QuantScheme::default().per_block([32], ScaleDtype::F32), 1);
    }

    #[test]
    fn a_two_level_scheme_takes_two_bindings() {
        check_scale_bindings(
            &QuantScheme::default()
                .per_block([32], ScaleDtype::F32)
                .per_tensor(ScaleDtype::F32),
            2,
        );
    }

    /// The binding is f32, so a level naming another dtype would have its scale read as f32 bytes.
    #[test]
    #[should_panic(expected = "binds as f32, but")]
    fn a_two_level_scheme_storing_the_tensor_scale_narrower_is_rejected() {
        check_scale_bindings(
            &QuantScheme::default()
                .per_block([32], ScaleDtype::F32)
                .per_tensor(ScaleDtype::BF16),
            2,
        );
    }

    #[test]
    #[should_panic(expected = "takes as many scale bindings, but 1 were provided")]
    fn a_two_level_scheme_with_one_binding_is_rejected() {
        // Would otherwise dequantize against the block scales alone, dropping the per-tensor factor.
        check_scale_bindings(
            &QuantScheme::default()
                .per_block([32], ScaleDtype::F32)
                .per_tensor(ScaleDtype::F32),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "takes as many scale bindings, but 2 were provided")]
    fn a_one_level_scheme_with_two_bindings_is_rejected() {
        check_scale_bindings(&QuantScheme::default().per_tensor(ScaleDtype::F32), 2);
    }
}
