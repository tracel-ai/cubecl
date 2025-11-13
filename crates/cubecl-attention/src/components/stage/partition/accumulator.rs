use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::tile::TileAttention;

use crate::components::stage::AccumulatorTile;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// Contains all seq_q·val_dim materialized tiles at once because they're accumulators
pub struct AccumulatorPartition<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> {
    sequence: Sequence<AccumulatorTile<AP, FA>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> AccumulatorPartition<AP, FA, S>
{
    pub fn new(#[comptime] config: S) -> AccumulatorPartition<AP, FA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.val_dim) {
            sequence.push(AccumulatorTile::new(config.tile_config()));
        }

        AccumulatorPartition::<AP, FA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &AccumulatorTile<AP, FA> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index(comptime!(i * p.val_dim + j))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &mut AccumulatorTile<AP, FA> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index_mut(comptime!(i * p.val_dim + j))
    }
}
