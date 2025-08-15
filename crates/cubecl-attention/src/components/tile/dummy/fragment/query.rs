use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::global::dummy::QueryRegisterReader;
use crate::components::tile::ScoreMatmul;

#[derive(CubeType)]
pub struct QueryFragment<AP: AttentionPrecision, SM: ScoreMatmul<AP>> {
    pub fragment: SM::Lhs,
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>> QueryFragment<AP, SM> {
    pub fn new(
        query_reader: QueryRegisterReader<AP>,
        #[comptime] config: SM::Config,
    ) -> QueryFragment<AP, SM> {
        let fragment = query_reader.read_tile::<SM>(config);
        QueryFragment::<AP, SM> { fragment }
    }
}
