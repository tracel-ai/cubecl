use cubecl_common::quant::scheme::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
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

/// Panic when the lookup-table binding and the scheme disagree.
///
/// The table binds as a buffer of its own, so nothing ties it to the mode: a missing one leaves
/// [`QuantMode::Lookup`] nothing to index, an extra one is a caller quantizing differently than
/// the scheme it passed. Lookup is also only wired where the decode goes through the packed-u32
/// unpack, and only for the integer values whose field is a plain bit range — a minifloat field
/// carries its own float semantics, which an index does not have.
///
/// The table must hold `2^bits` f32 entries; a shorter one is read out of bounds, which no check
/// here can see (the binding's length is runtime).
pub fn check_table_bindings(scheme: &QuantScheme, table_provided: bool) {
    match (scheme.mode, table_provided) {
        (QuantMode::Lookup, false) => {
            panic!(
                "{:?} takes a lookup table, but none was provided",
                scheme.mode
            )
        }
        (QuantMode::Lookup, true) => {
            assert!(
                matches!(scheme.store, QuantStore::PackedU32(_)),
                "lookup decode is only wired for packed-u32 storage, got {:?}",
                scheme.store
            );
            assert!(
                !matches!(
                    scheme.value,
                    QuantValue::E5M2 | QuantValue::E4M3 | QuantValue::E2M1
                ),
                "a lookup field is an index, so a minifloat value ({:?}) has nothing to mean; \
                 use the integer value of the same width",
                scheme.value
            );
        }
        (_, true) => {
            panic!(
                "a lookup table was provided, but {:?} does not take one",
                scheme.mode
            )
        }
        (_, false) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{check_global_bindings, check_table_bindings};
    use cubecl_common::quant::scheme::{
        QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
    };

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

    fn lookup_scheme() -> QuantScheme {
        QuantScheme::default()
            .with_value(QuantValue::Q4F)
            .with_mode(QuantMode::Lookup)
    }

    #[test]
    fn a_lookup_scheme_takes_a_table() {
        check_table_bindings(&lookup_scheme(), true);
    }

    #[test]
    fn a_symmetric_scheme_takes_no_table() {
        check_table_bindings(&QuantScheme::default(), false);
    }

    #[test]
    #[should_panic(expected = "takes a lookup table, but none was provided")]
    fn a_lookup_scheme_without_a_table_is_rejected() {
        // Would otherwise fall back to the integer cast and reconstruct the index itself.
        check_table_bindings(&lookup_scheme(), false);
    }

    #[test]
    #[should_panic(expected = "does not take one")]
    fn a_symmetric_scheme_with_a_table_is_rejected() {
        check_table_bindings(&QuantScheme::default(), true);
    }

    /// Only the packed-u32 unpack decodes through the table; the native paths cast the storage
    /// element directly and would silently ignore it.
    #[test]
    #[should_panic(expected = "only wired for packed-u32 storage")]
    fn a_native_lookup_scheme_is_rejected() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8F)
            .with_store(QuantStore::Native)
            .with_mode(QuantMode::Lookup);
        check_table_bindings(&scheme, true);
    }

    #[test]
    #[should_panic(expected = "a lookup field is an index")]
    fn a_minifloat_lookup_scheme_is_rejected() {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E4M3)
            .with_mode(QuantMode::Lookup);
        check_table_bindings(&scheme, true);
    }
}
