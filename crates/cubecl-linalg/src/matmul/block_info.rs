use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockInfos {
    pub lhs: BlockInfo,
    pub rhs: BlockInfo,
    pub out: BlockInfo,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockInfo {
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub tile_size_x: u32,
    pub tile_size_y: u32,
}

#[cube]
pub fn total_num_elements(#[comptime] block_info: BlockInfo) -> u32 {
    comptime!(
        block_info.num_tiles_x
            * block_info.num_tiles_y
            * block_info.tile_size_x
            * block_info.tile_size_y
    )
}

#[cube]
pub fn tile_num_elements(#[comptime] block_info: BlockInfo) -> u32 {
    comptime!(block_info.tile_size_x * block_info.tile_size_y)
}

impl CubeType for BlockInfos {
    type ExpandType = Self;
}

impl Init for BlockInfos {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl IntoRuntime for BlockInfos {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        self
    }
}

impl CubeType for BlockInfo {
    type ExpandType = Self;
}

impl Init for BlockInfo {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl IntoRuntime for BlockInfo {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        self
    }
}
