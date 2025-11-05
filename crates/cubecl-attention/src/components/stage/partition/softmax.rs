use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::fragment::FragmentAttention;
use crate::components::tile::SoftmaxTile;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};

#[derive(CubeType)]
/// Because at each hd we will perform matmul with all of seq_q, we keep seq_q softmax tiles at a time.
/// Each of the seq_kv column can be done sequentially reusing those tiles.
pub struct SoftmaxPartition<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> {
    sequence: Sequence<SoftmaxTile<AP, FA>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> SoftmaxPartition<AP, FA, S>
{
    pub fn new(#[comptime] config: S) -> SoftmaxPartition<AP, FA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q) {
            sequence.push(SoftmaxTile::new(config.tile_config()));
        }

        SoftmaxPartition::<AP, FA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(&self, #[comptime] q: u32, #[comptime] config: S) -> &SoftmaxTile<AP, FA> {
        self.sequence.index(q)
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] config: S,
    ) -> &mut SoftmaxTile<AP, FA> {
        self.sequence.index_mut(q)
    }
}
