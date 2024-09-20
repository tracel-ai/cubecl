use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Ids {
    pub coop: u32,
    pub lane: u32,
}

#[cube]
pub(crate) trait IdDispatch {
    fn get_compute_ids() -> Ids;
    fn get_load_ids() -> Ids;
}

pub(crate) struct SameRoleIdDispatch {}

#[cube]
impl IdDispatch for SameRoleIdDispatch {
    fn get_compute_ids() -> Ids {
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }

    fn get_load_ids() -> Ids {
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }
}
