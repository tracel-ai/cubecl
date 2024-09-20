use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Ids {
    pub coop: u32,
    pub lane: u32,
}

#[cube]
pub(crate) fn get_ids() -> (Ids, Ids) {
    (
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        },
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        },
    )
}
