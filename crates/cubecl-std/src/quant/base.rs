use cubecl_common::quant::scheme::{BlockSize, QuantScheme, ScaleDtype};
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
/// A coarser level binds as f32 rather than being read as f32 bytes. f32 is required only because
/// every coarser level in use is f32; another dtype is its binding read as that dtype, as the
/// finest level's is.
pub fn check_scale_bindings(scheme: &QuantScheme, bindings: usize) {
    let levels = scheme.levels().count();
    assert!(
        bindings == levels,
        "a scheme with {levels} scale level(s) takes as many scale bindings, but {bindings} were provided",
    );
    for coarser in scheme.levels().skip(1) {
        assert!(
            coarser.dtype == ScaleDtype::F32,
            "a coarser scale binds as f32, but the scheme stores it as {:?}",
            coarser.dtype
        );
    }
}

/// Panic where a consumer holds the coarser scale in a register, one value for every position,
/// and the scheme's coarser level is not one f32 scale for the whole tensor.
///
/// # Panics
///
/// Where a level coarser than the finest is not one f32 scale for the whole tensor.
pub fn check_global_levels(scheme: &QuantScheme) {
    for global in scheme.levels().skip(1) {
        assert!(
            global.size == BlockSize::TENSOR,
            "a global scale is one for the whole tensor, but the scheme has one per {:?}",
            global.size
        );
        assert!(
            global.dtype == ScaleDtype::F32,
            "a global scale binds as f32, but the scheme stores it as {:?}",
            global.dtype
        );
    }
}

/// Panic when the lookup-table binding and the scheme disagree.
///
/// The table binds as a buffer of its own, so nothing ties it to the mode: a missing one leaves
/// [`QuantMode::Lookup`](cubecl_common::quant::scheme::QuantMode) nothing to index, an extra one
/// is a caller quantizing differently than the scheme it passed. Lookup is also only wired where
/// the decode goes through the packed-u32 unpack, and only for the integer values whose field is
/// a plain bit range — a minifloat field carries its own float semantics, which an index does not
/// have.
///
/// The table must hold `2^bits` f32 entries; [`register_table`](crate::quant::view) checks the
/// binding's length against that, the one host-side site holding both.
pub fn check_table_bindings(scheme: &QuantScheme, table_provided: bool) {
    use cubecl_common::quant::scheme::{QuantMode, QuantStore, QuantValue};
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
    use super::{check_global_levels, check_scale_bindings, check_table_bindings};
    use cubecl_common::quant::scheme::{
        QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype,
    };

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
    fn a_coarser_level_of_rows_binds_as_a_grid() {
        check_scale_bindings(&QuantScheme::mxfp4_row_factored(), 2);
    }

    /// A reader holding the coarser scale in a register has one value for every position.
    #[test]
    #[should_panic(expected = "is one for the whole tensor, but")]
    fn a_coarser_level_of_rows_is_no_register_scale() {
        check_global_levels(&QuantScheme::mxfp4_row_factored());
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
