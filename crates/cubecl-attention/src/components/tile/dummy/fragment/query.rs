use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::dummy::AttentionMatmul;
use crate::components::tile::{QueryTile, QueryTileExpand};
use cubecl_matmul::components::tile::StridedTile;

#[derive(CubeType)]
pub struct QueryFragment<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Query,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> QueryFragment<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> QueryFragment<AP, AM> {
        QueryFragment::<AP, AM> {
            fragment: AM::allocate_query(config),
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> QueryTile<AP> for QueryFragment<AP, AM> {
    type Fragment = AM::Query;

    fn fragment_mut(&mut self) -> &mut Self::Fragment {
        &mut self.fragment
    }

    fn update(&mut self, tile: StridedTile<QG<AP>>) {
        AM::fill_query(&tile, &mut self.fragment)
    }
}
