use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matmul_stage::{TilingOrder, XMajorTiling};
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
// TODO rename to Stage
pub struct SharedMemoryStage<ES: Numeric> {
    pub smem: SharedMemory<Line<ES>>,
}

#[cube]
pub(crate) fn new_stage<ES: Numeric, S: SmmConfig>(
    #[comptime] ident: Ident,
    #[comptime] config: S,
) -> SharedMemoryStage<ES> {
    let smem = SharedMemory::new_lined(
        comptime!(config.stage_dim(ident).num_elements() / config.line_size(ident)),
        config.line_size(ident),
    );

    SharedMemoryStage::<ES> { smem }
}

#[cube]
pub(crate) fn get_tile<ES: Numeric, S: SmmConfig>(
    this: &SharedMemoryStage<ES>,
    x: u32,
    y: u32,
    #[comptime] ident: Ident,
    #[comptime] config: S,
) -> &Slice<'_, Line<ES>> {
    let stage_dim = config.stage_dim(ident);

    // TODO X or Y choose with comptime config
    let nth_tile = XMajorTiling::to_nth_tile(x, y, stage_dim.num_tiles_x, stage_dim.num_tiles_y);

    let tile_stride = stage_dim.tile_num_elements() / config.line_size(ident);
    let start = nth_tile * tile_stride;

    this.smem.slice(start, start + tile_stride)
}
