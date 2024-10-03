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

create_cmma_block!(S128_128_16, 128, 128, 16);
create_cmma_block!(S16_16_16, 16, 16, 16);
create_cmma_block!(S32_8_16, 32, 8, 16);
create_cmma_block!(S8_32_16, 8, 32, 16);
create_cmma_block!(S32_16_16, 32, 16, 16);
create_cmma_block!(S128_16_16, 128, 16, 16);
create_cmma_block!(S32_32_16, 32, 32, 16);
create_cmma_block!(S64_64_16, 64, 64, 16);
create_cmma_block!(S32_32_32, 32, 32, 32);

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

#[derive(CubeType, Clone, Copy)]
pub struct BlockInfoR {
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub tile_size_x: u32,
    pub tile_size_y: u32,
}

#[cube]
pub fn num_elements(#[comptime] block_info: BlockInfo) -> u32 {
    comptime!(
        block_info.num_tiles_x
            * block_info.num_tiles_y
            * block_info.tile_size_x
            * block_info.tile_size_y
    )
}

#[cube]
pub fn into_runtime(#[comptime] comptime: BlockInfo) -> BlockInfoR {
    BlockInfoR {
        num_tiles_x: comptime.num_tiles_x,
        num_tiles_y: comptime.num_tiles_y,
        tile_size_x: comptime.tile_size_x,
        tile_size_y: comptime.tile_size_y,
    }
}
