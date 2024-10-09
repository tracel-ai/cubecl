use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct StageInfos {
    pub lhs: StageInfo,
    pub rhs: StageInfo,
    pub out: StageInfo,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct StageInfo {
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub tile_size_x: u32,
    pub tile_size_y: u32,
}

#[cube]
pub fn total_num_elements(#[comptime] block_info: StageInfo) -> u32 {
    comptime!(
        block_info.num_tiles_x
            * block_info.num_tiles_y
            * block_info.tile_size_x
            * block_info.tile_size_y
    )
}

#[cube]
pub fn tile_num_elements(#[comptime] block_info: StageInfo) -> u32 {
    comptime!(block_info.tile_size_x * block_info.tile_size_y)
}

impl CubeType for StageInfos {
    type ExpandType = Self;
}

impl Init for StageInfos {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl IntoRuntime for StageInfos {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        self
    }
}

impl CubeType for StageInfo {
    type ExpandType = Self;
}

impl Init for StageInfo {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl IntoRuntime for StageInfo {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        self
    }
}
