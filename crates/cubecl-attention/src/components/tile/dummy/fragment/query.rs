use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::tile::dummy::{FlashMatmul, FlashPrecision};

#[derive(CubeType)]
pub struct QueryFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    pub fragment: FM::Query,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> QueryFragment<FP, FM> {
    pub fn new<E: Numeric>(
        tile: &StridedTile<E>,
        #[comptime] config: FM::Config,
    ) -> QueryFragment<FP, FM> {
        QueryFragment::<FP, FM> {
            fragment: FM::allocate_fill_query(tile, config),
        }
    }
}
