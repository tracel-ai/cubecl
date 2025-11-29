use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::tile::TileAttention;

use crate::components::stage::{AccumulatorTile, PartitionAttentionConfig};
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// Contains all seq_q·val_dim materialized tiles at once because they're accumulators
pub struct AccumulatorPartition<AP: AttentionPrecision, TA: TileAttention<AP>> {
    sequence: Sequence<AccumulatorTile<AP, TA>>,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> AccumulatorPartition<AP, TA> {
    pub fn new(
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> AccumulatorPartition<AP, TA> {
        let p = config.shared().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q * p.val_dim) {
            sequence.push(AccumulatorTile::new(config.tile_config()));
        }

        AccumulatorPartition::<AP, TA> { sequence }
    }

    pub fn get_at(
        &self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> &AccumulatorTile<AP, TA> {
        let partition_val_dim = config.shared().partition_size.val_dim;
        self.sequence.index(comptime!(i * partition_val_dim + j))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> &mut AccumulatorTile<AP, TA> {
        let partition_val_dim = config.shared().partition_size.val_dim;
        self.sequence
            .index_mut(comptime!(i * partition_val_dim + j))
    }
}
