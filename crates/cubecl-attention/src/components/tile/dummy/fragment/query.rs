use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::global::dummy::QueryRegisterReader;
use crate::components::tile::dummy::{FlashMatmul, FlashPrecision};

#[derive(CubeType)]
pub struct QueryFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    pub fragment: FM::Query,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> QueryFragment<FP, FM> {
    pub fn new<E: Numeric>(
        query_reader: QueryRegisterReader<E>,
        #[comptime] config: FM::Config,
    ) -> QueryFragment<FP, FM> {
        comment!("Reading query");
        let fragment = query_reader.read_tile::<FP, FM>(config);
        QueryFragment::<FP, FM> { fragment }
    }
}
