use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::TileAttention;
use cubecl_matmul::components::tile::StridedTile;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct QueryTile<AP: AttentionPrecision, FA: TileAttention<AP>> {
    pub fragment: FA::Query,
}

#[cube]
impl<AP: AttentionPrecision, FA: TileAttention<AP>> QueryTile<AP, FA> {
    pub fn new(#[comptime] config: FA::Config) -> QueryTile<AP, FA> {
        QueryTile::<AP, FA> {
            fragment: FA::allocate_query(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update(&mut self, tile: &StridedTile<QG<AP>>) {
        FA::fill_query(tile, &mut self.fragment)
    }
}
