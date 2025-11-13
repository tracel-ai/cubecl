use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// Because at each hd we will perform matmul with all of seq_q, we keep seq_q softmax tiles at a time.
/// Each of the seq_kv column can be done sequentially reusing those tiles.
pub struct SoftmaxPartition<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> {
    sequence: Sequence<FA::Softmax>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> SoftmaxPartition<AP, FA, S>
{
    pub fn new(#[comptime] config: S) -> SoftmaxPartition<AP, FA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q) {
            sequence.push(FA::allocate_softmax(config.tile_config()));
        }

        SoftmaxPartition::<AP, FA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(&self, #[comptime] q: u32) -> &FA::Softmax {
        self.sequence.index(q)
    }

    pub fn get_at_mut(&mut self, #[comptime] q: u32) -> &mut FA::Softmax {
        self.sequence.index_mut(q)
    }
}
