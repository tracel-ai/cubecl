use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::AttentionTilingScheme;
use crate::components::fragment::FragmentAttention;
use crate::components::tile::MaskTile;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType)]
/// We can keep only one mask tile at a time because it is directly applied to softmax tile
pub struct MaskPartition<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> {
    sequence: Sequence<MaskTile<AP, FA>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> MaskPartition<AP, FA, S>
{
    pub fn new(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: S,
    ) -> MaskPartition<AP, FA, S> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        let mut q = comptime![0];

        #[unroll]
        for _ in 0..p.seq_q {
            let mut kv = comptime![0];

            #[unroll]
            for _ in 0..p.seq_kv {
                sequence.push(MaskTile::new(out_of_bounds, (q, kv), config.tile_config()));

                comptime![kv += 1];
            }

            comptime![q += 1];
        }

        MaskPartition::<AP, FA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] tiling_scheme: AttentionTilingScheme,
    ) -> &MaskTile<AP, FA> {
        let p = tiling_scheme.partition_size;
        self.sequence.index(comptime!(q * p.seq_kv + kv))
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] q: u32,
        #[comptime] kv: u32,
        #[comptime] tiling_scheme: AttentionTilingScheme,
    ) -> &mut MaskTile<AP, FA> {
        let p = tiling_scheme.partition_size;
        self.sequence.index_mut(comptime!(q * p.seq_kv + kv))
    }
}
