use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::stage::{MaskTile, PartitionAttentionConfig};
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, stage::StageAttentionConfig};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType)]
/// We can keep only one mask tile at a time because it is directly applied to softmax tile
pub struct MaskPartition<AP: AttentionPrecision, TA: TileAttention<AP>> {
    sequence: Sequence<MaskTile<AP, TA>>,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> MaskPartition<AP, TA> {
    pub fn new(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> MaskPartition<AP, TA> {
        let mut sequence = Sequence::new();

        sequence.push(MaskTile::new(out_of_bounds, config.tile_config()));

        MaskPartition::<AP, TA> { sequence }
    }

    pub fn get(&self) -> &MaskTile<AP, TA> {
        self.sequence.index(0u32)
    }

    pub fn get_mut(&mut self) -> &mut MaskTile<AP, TA> {
        self.sequence.index_mut(0u32)
    }
}
