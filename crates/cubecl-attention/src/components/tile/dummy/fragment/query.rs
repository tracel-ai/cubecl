use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::AttentionPrecision;
use crate::components::attention_types::QT;
use crate::components::tile::QueryTile;
use crate::components::tile::dummy::AttentionMatmul;

#[derive(CubeType)]
pub struct QueryFragment<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Query,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> QueryFragment<AP, AM> {
    pub fn new<E: Float>(
        tile: &StridedTile<E>,
        #[comptime] config: AM::Config,
    ) -> QueryFragment<AP, AM> {
        QueryFragment::<AP, AM> {
            fragment: AM::allocate_fill_query(tile, config),
        }
    }
}

impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> QueryTile<QT<AP>> for QueryFragment<AP, AM> {}
