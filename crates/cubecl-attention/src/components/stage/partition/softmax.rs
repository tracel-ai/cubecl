use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::stage::PartitionAttentionConfig;
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// Because at each hd we will perform matmul with all of seq_q, we keep seq_q softmax tiles at a time.
/// Each of the seq_kv column can be done sequentially reusing those tiles.
pub struct SoftmaxPartition<AP: AttentionPrecision, TA: TileAttention<AP>> {
    sequence: Sequence<TA::Softmax>,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> SoftmaxPartition<AP, TA> {
    pub fn new(
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> SoftmaxPartition<AP, TA> {
        let p = config.shared().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q) {
            sequence.push(TA::allocate_softmax(config.tile_config()));
        }

        SoftmaxPartition::<AP, TA> { sequence }
    }

    pub fn get_at(&self, #[comptime] q: u32) -> &TA::Softmax {
        self.sequence.index(q)
    }

    pub fn get_at_mut(&mut self, #[comptime] q: u32) -> &mut TA::Softmax {
        self.sequence.index_mut(q)
    }
}
