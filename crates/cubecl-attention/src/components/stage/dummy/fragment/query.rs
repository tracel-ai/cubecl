use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::TileMatmul;

use crate::components::AttentionPrecision;
use crate::components::global::dummy::QueryRegisterReader;

#[derive(CubeType)]
pub struct QueryFragment<AP: AttentionPrecision, STM: TileMatmul<AP::MatmulPrecision>> {
    pub fragment: STM::Lhs,
}

#[cube]
impl<AP: AttentionPrecision, STM: TileMatmul<AP::MatmulPrecision>> QueryFragment<AP, STM> {
    pub fn new(
        query_reader: QueryRegisterReader<AP>,
        #[comptime] config: STM::Config,
    ) -> QueryFragment<AP, STM> {
        let fragment = query_reader.read_tile::<STM>(config);
        QueryFragment::<AP, STM> { fragment }
    }
}
