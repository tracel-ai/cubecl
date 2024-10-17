use crate::matmul::cmma_matmul::stage::CmmaStageMatmulConfig;
use crate::matmul::matmul_global::ReadView;
use crate::matmul::matmul_stage::{TilingOrder, XMajorTiling};
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType, Clone, Copy)]
// TODO rename to Stage
pub struct SharedMemoryStage<ES: Numeric> {
    smem: SharedMemory<Line<ES>>,
}

#[cube]
fn new(#[comptime] ident: Ident, #[comptime] config: CmmaStageMatmulConfig<T>) -> Self {
    let smem = SharedMemory::new_lined(
        comptime!(config.stage_num_elems(ident) / config.line_size(ident)),
        config.line_size(ident),
    );

    SharedMemoryStage::<ES, O> {
        smem,
        _tiling_order: PhantomData::<O>.runtime(),
    }
}

#[cube]
fn fill<EG: Numeric, ES: Numeric, RV: ReadView<EG>, T: TmmConfig>(
    self_: &mut SharedMemoryStage,
    read_view: &RV,
    #[comptime] ident: Ident,
    #[comptime] config: CmmaStageMatmulConfig<T>,
) {
    RV::load_shared_memory::<ES>(read_view, &mut self_.smem, ident, config)
}

#[cube]
fn get_tile(
    self_: &SharedMemoryStage,
    x: u32,
    y: u32,
    #[comptime] ident: Ident,
    #[comptime] config: Self::Config,
) -> &Slice<'_, Line<ES>> {
    let stage_dim = config.stage_dim(ident);

    // TODO X or Y choose with comptime config
    let nth_tile = XMajorTiling::to_nth_tile(x, y, stage_dim.num_tiles_x, stage_dim.num_tiles_y);

    let tile_stride = stage_dim.tile_num_elements() / config.line_size(ident);
    let start = nth_tile * tile_stride;

    self_.smem.slice(start, start + tile_stride)
}
