use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::AttentionMatmul;
use cubecl_matmul::components::tile::StridedTile;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct QueryTile<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Query,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> QueryTile<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> QueryTile<AP, AM> {
        QueryTile::<AP, AM> {
            fragment: AM::allocate_query(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update(&mut self, tile: &StridedTile<QG<AP>>) {
        AM::fill_query(tile, &mut self.fragment)
    }
}
