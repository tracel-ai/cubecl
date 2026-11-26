use crate::quant::scheme::{
    BlockExtent, BlockScale, BlockSize, QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype,
};

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
            .per_block(
                BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE),
                ScaleDtype::F32,
            )
    }

    /// MXFP4, row-factored scales: fp4 (`e2m1`) values, eight to a word along the last axis, in
    /// blocks of 32 under `ue8m0` block scales, factored by row. The values are MXFP4's own: each
    /// row's scale, an `f32`, is its largest block scale, and each block scale is stored as its
    /// offset below that one, a byte a block as before. Factoring adds no precision; it puts each
    /// row's largest block scale at one, so a decode into `e4m3` takes the block scales in exactly.
    pub fn mxfp4_row_factored() -> Self {
        QuantScheme {
            value: QuantValue::E2M1,
            store: QuantStore::PackedU32(0),
            mode: QuantMode::Symmetric,
            blocks: [
                Some(BlockScale {
                    size: BlockSize::new([32]),
                    dtype: ScaleDtype::UE8M0,
                }),
                Some(BlockScale {
                    size: BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE),
                    dtype: ScaleDtype::F32,
                }),
            ],
        }
    }
}
