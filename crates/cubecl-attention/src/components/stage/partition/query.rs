use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::stage::QueryTile;
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// Contains all seq_q·head_dim materialized tiles at once because they are reused extensively
pub struct QueryPartition<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> {
    sequence: Sequence<QueryTile<AP, FA>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> QueryPartition<AP, FA, S>
{
    pub fn new(#[comptime] config: S) -> QueryPartition<AP, FA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.head_dim) {
            sequence.push(QueryTile::<AP, FA>::new(config.tile_config()));
        }

        QueryPartition::<AP, FA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: S,
    ) -> &QueryTile<AP, FA> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index(comptime!(q * p.head_dim + hd))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: S,
    ) -> &mut QueryTile<AP, FA> {
        let p = config.tiling_scheme().partition_size;
        self.sequence.index_mut(comptime!(q * p.head_dim + hd))
    }
}
