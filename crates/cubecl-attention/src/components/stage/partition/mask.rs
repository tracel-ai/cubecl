use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::stage::MaskTile;
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType)]
/// We can keep only one mask tile at a time because it is directly applied to softmax tile
pub struct MaskPartition<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> {
    sequence: Sequence<MaskTile<AP, FA>>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    FA: TileAttention<AP>,
    S: StageAttentionConfig<TileAttentionConfig = FA::Config>,
> MaskPartition<AP, FA, S>
{
    pub fn new(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: S,
    ) -> MaskPartition<AP, FA, S> {
        let mut sequence = Sequence::new();

        sequence.push(MaskTile::new(out_of_bounds, config.tile_config()));

        MaskPartition::<AP, FA, S> {
            sequence,
            _phantom: PhantomData,
        }
    }

    pub fn get(&self) -> &MaskTile<AP, FA> {
        self.sequence.index(0u32)
    }

    pub fn get_mut(&mut self) -> &mut MaskTile<AP, FA> {
        self.sequence.index_mut(0u32)
    }
}
