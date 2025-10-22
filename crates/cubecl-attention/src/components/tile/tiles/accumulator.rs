use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::AttentionMatmul;
use crate::components::fragment::{FragmentOps, FragmentOpsExpand};
use crate::components::tile::RowWise;

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct AccumulatorTile<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Accumulator,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> AccumulatorTile<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> AccumulatorTile<AP, AM> {
        let mut fragment = AM::allocate_accumulator(config);
        AM::zero_accumulator(&mut fragment);

        AccumulatorTile::<AP, AM> { fragment }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> AccumulatorTile<AP, AM> {
    /// Multiplies each row by a scale
    pub fn scale_mul(&mut self, scale: &RowWise<SM<AP>>) {
        self.fragment
            .rowwise_scale(&RowWise::<SM<AP>>::cast_from(scale));
    }

    /// Divides each row by a scale
    pub fn scale_div(&mut self, scale: &RowWise<SM<AP>>) {
        let mut scale = RowWise::<SM<AP>>::cast_from(scale);
        scale.recip_inplace();
        self.fragment.rowwise_scale(&scale);
    }
}
