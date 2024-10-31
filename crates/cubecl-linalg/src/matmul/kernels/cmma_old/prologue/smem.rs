use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::super::config::ComptimeCmmaInfo;

#[derive(CubeType, Copy, Clone)]
pub(crate) struct SharedMemories<FC: Float> {
    pub lhs: SharedMemory<FC>,
    pub rhs: SharedMemory<FC>,
}

#[cube]
pub(crate) fn make_shared_memories<FC: Float>(
    #[comptime] config: ComptimeCmmaInfo,
) -> SharedMemories<FC> {
    let block_size_m = config.block_size_m;
    let block_size_k = config.block_size_k;
    let block_size_n = config.block_size_n;

    let lhs = SharedMemory::<FC>::new(block_size_k * block_size_m);
    let rhs = SharedMemory::<FC>::new(block_size_k * block_size_n);

    SharedMemories::<FC> { lhs, rhs }
}
