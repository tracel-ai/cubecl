use cubecl_core as cubecl;
use cubecl_core::{CubeLaunch, CubeType, prelude::*};
use cubecl_std::FastDivmod;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub shape_m: u32,
    pub shape_n: u32,
    pub shape_k: u32,
    pub padded_channels: FastDivmod,
    pub shape_out: Sequence<FastDivmod>,
    pub shape_channel: FastDivmod,
}
