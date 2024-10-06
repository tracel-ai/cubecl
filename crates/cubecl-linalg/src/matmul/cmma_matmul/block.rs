use cubecl_core as cubecl;
use cubecl_core::prelude::*;

pub trait CmmaBlockSize: 'static + Send + Sync {
    const M: u32;
    const N: u32;
    const K: u32;
}

macro_rules! create_cmma_block {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name;

        impl CmmaBlockSize for $name {
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;
        }
    };
}

create_cmma_block!(B128x128x16, 128, 128, 16);
create_cmma_block!(B16x16x16, 16, 16, 16);
create_cmma_block!(B32x8x16, 32, 8, 16);
create_cmma_block!(B8x32x16, 8, 32, 16);
create_cmma_block!(B32x16x16, 32, 16, 16);
create_cmma_block!(B128x16x16, 128, 16, 16);
create_cmma_block!(B32x32x16, 32, 32, 16);
create_cmma_block!(B64x64x16, 64, 64, 16);
create_cmma_block!(B32x32x32, 32, 32, 32);
create_cmma_block!(B64x64x32, 64, 64, 32);
create_cmma_block!(B16x32x16, 16, 32, 16);

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
