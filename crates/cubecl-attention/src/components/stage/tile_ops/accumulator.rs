use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::RowWise;
use crate::components::tile::TileAttention;
use crate::components::tile::{FragmentAccumulator, FragmentAccumulatorExpand};

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct AccumulatorTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    pub fragment: TA::Accumulator,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> AccumulatorTile<AP, TA> {
    pub fn new(#[comptime] config: TA::Config) -> AccumulatorTile<AP, TA> {
        let mut fragment = TA::allocate_accumulator(config);
        fragment.zero();

        AccumulatorTile::<AP, TA> { fragment }
    }
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> AccumulatorTile<AP, TA> {
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
