use cubecl_core as cubecl;
use cubecl_core::{CubeLaunch, CubeType, prelude::*};
use cubecl_std::FastDivmod;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub size_m: u32,
    pub size_n: u32,
    pub size_k: u32,
    pub padded_channels: FastDivmod,
    pub out_shape: Sequence<FastDivmod>,
}
