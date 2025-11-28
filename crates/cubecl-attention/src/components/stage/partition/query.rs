use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::stage::{PartitionAttentionConfig, QueryTile};
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// Contains all seq_q·head_dim materialized tiles at once because they are reused extensively
pub struct QueryPartition<AP: AttentionPrecision, TA: TileAttention<AP>> {
    sequence: Sequence<QueryTile<AP, TA>>,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> QueryPartition<AP, TA> {
    pub fn new(#[comptime] config: PartitionAttentionConfig<TA::Config>) -> QueryPartition<AP, TA> {
        let p = config.shared().partition_size;

        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.head_dim) {
            sequence.push(QueryTile::<AP, TA>::new(config.tile_config()));
        }

        QueryPartition::<AP, TA> { sequence }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> &QueryTile<AP, TA> {
        let partition_head_dim = config.shared().partition_size.head_dim;
        self.sequence.index(comptime!(q * partition_head_dim + hd))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] hd: u32,
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> &mut QueryTile<AP, TA> {
        let partition_head_dim = config.shared().partition_size.head_dim;
        self.sequence
            .index_mut(comptime!(q * partition_head_dim + hd))
    }
}
