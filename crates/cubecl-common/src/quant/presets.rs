use crate::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};

impl QuantScheme {
    /// The NVFP4 format: fp4 (e2m1) values in blocks of 16, with ue4m3 block scales
    /// normalized by one per-tensor f32 scale.
    pub fn nvfp4() -> Self {
        QuantScheme::per_block([16], ScaleDtype::UE4M3)
            .and_per_tensor(ScaleDtype::F32)
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::PackedNative(0))
    }
}
