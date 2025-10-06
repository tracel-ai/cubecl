use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::AccumulatorTile;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::RowWise;
use crate::components::tile::dummy::AttentionMatmul;
use crate::components::tile::row::{PlaneLayout, PlaneLayoutExpand};

#[derive(CubeType)]
pub struct DummyAccumulator<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Accumulator,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> DummyAccumulator<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> DummyAccumulator<AP, AM> {
        let mut fragment = AM::allocate_accumulator(config);
        AM::zero_accumulator(&mut fragment);

        DummyAccumulator::<AP, AM> { fragment }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> AccumulatorTile<AP>
    for DummyAccumulator<AP, AM>
{
    fn scale_mul(&mut self, scale: &RowWise<SM<AP>>) {
        self.fragment.scale(&RowWise::<SM<AP>>::cast_from(scale));
    }

    fn scale_div(&mut self, scale: &RowWise<SM<AP>>) {
        let mut scale = RowWise::<SM<AP>>::cast_from(scale);
        scale.recip_inplace();
        self.fragment.scale(&scale);
    }
}
