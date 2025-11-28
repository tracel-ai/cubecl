use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::TileAttention;
use cubecl_matmul::components::tile::StridedTile;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct QueryTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    pub fragment: TA::Query,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> QueryTile<AP, TA> {
    pub fn new(#[comptime] config: TA::Config) -> QueryTile<AP, TA> {
        QueryTile::<AP, TA> {
            fragment: TA::allocate_query(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update(&mut self, tile: &StridedTile<QG<AP>>) {
        TA::load_query(tile, &mut self.fragment)
    }
}
