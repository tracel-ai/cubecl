use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::AccumulatorTile;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::dummy::AttentionMatmul;
use crate::components::tile::row::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowWise, RowWiseExpand};

#[derive(CubeType)]
pub struct DummyAccumulator<AP: AttentionPrecision, AM: AttentionMatmul<AP>, RW: RowWise> {
    pub fragment: AM::Accumulator,

    #[cube(comptime)]
    _phantom: PhantomData<RW>,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>, RW: RowWise> DummyAccumulator<AP, AM, RW> {
    pub fn new(#[comptime] config: AM::Config) -> DummyAccumulator<AP, AM, RW> {
        let mut fragment = AM::allocate_accumulator(config);
        AM::zero_accumulator(&mut fragment);

        DummyAccumulator::<AP, AM, RW> {
            fragment,
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>, RW: RowWise> AccumulatorTile<AP, RW>
    for DummyAccumulator<AP, AM, RW>
{
    fn scale_mul(&mut self, scale: &RW) {
        self.fragment.scale(ACC::<AP>::cast_from(scale.index(0u32)));
    }

    fn scale_div(&mut self, scale: &RW) {
        self.fragment
            .scale(Recip::recip(ACC::<AP>::cast_from(scale.index(0u32))));
    }
}
