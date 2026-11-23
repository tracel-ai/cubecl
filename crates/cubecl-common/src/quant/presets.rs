use crate::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};

impl QuantScheme {
    /// The NVFP4 format: fp4 (e2m1) values in blocks of 16, with ue4m3 block scales
    /// normalized by one per-tensor f32 scale.
    pub fn nvfp4() -> Self {
        QuantScheme::default()
            .per_block([16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32)
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::PackedNative(0))
    }

    /// FP8 with one scale per row: `e4m3` values, four to a word along the last axis, under one
    /// `f32` scale for each row, over the whole last axis.
    pub fn fp8_per_row() -> Self {
        QuantScheme::default()
            .with_value(QuantValue::E4M3)
            .with_store(QuantStore::PackedU32(0))
            .per_row(ScaleDtype::F32)
    }

    /// MXFP4 under one scale per row (MXFP4-row): fp4 (`e2m1`) values, eight to a word along the
    /// last axis, in blocks of 32 under `ue8m0` block scales, themselves under one `f32` scale for
    /// each row. An MXFP4 checkpoint's rows restated under their largest block scale have their
    /// block scales stored as offsets below it, one byte a block as before, and that scale as the
    /// row's: every value is unchanged, and each block's offset is at most one.
    pub fn mxfp4_per_row() -> Self {
        QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::PackedU32(0))
            .per_block([32], ScaleDtype::UE8M0)
            .per_row(ScaleDtype::F32)
    }
}
