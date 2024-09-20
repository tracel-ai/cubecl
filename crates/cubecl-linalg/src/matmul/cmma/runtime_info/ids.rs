use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::super::config::ComptimeCmmaInfo;

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Ids {
    pub coop: u32,
    pub lane: u32,
}

#[cube]
pub(crate) trait IdDispatch {
    fn split_roles() -> bool;
    fn get_compute_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids;
    fn get_load_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids;
}

pub(crate) struct SameRoleIdDispatch {}
pub(crate) struct SplitRolesHalfwayIdDispatch {}

#[cube]
impl IdDispatch for SameRoleIdDispatch {
    fn split_roles() -> bool {
        false
    }

    fn get_compute_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }

    fn get_load_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }
}

#[cube]
/// Assumes CUBE_DIM_Y = comptime_info.num_compute_coops + comptime_info.num_load_coops
impl IdDispatch for SplitRolesHalfwayIdDispatch {
    fn split_roles() -> bool {
        true
    }

    fn get_compute_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }

    fn get_load_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids {
        // offset by number of compute coops
        Ids {
            coop: UNIT_POS_Y + comptime_info.num_compute_coops,
            lane: UNIT_POS_X,
        }
    }
}
