use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::matmul_global::ReadView;
use crate::matmul::matmul_stage::{Stage, TilingOrder};
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryStage<E: Numeric, O: TilingOrder> {
    smem: SharedMemory<Line<E>>,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<ES: Numeric, O: TilingOrder> Stage<ES> for SharedMemoryStage<ES, O> {
    type Underlying = SharedMemory<Line<ES>>;
    type Config = CmmaConfig;

    fn new(#[comptime] ident: Ident, #[comptime] config: Self::Config) -> Self {
        let smem = SharedMemory::new_lined(
            comptime!(config.stage_num_elems(ident) / config.line_size(ident)),
            config.line_size(ident),
        );

        SharedMemoryStage::<ES, O> {
            smem,
            _tiling_order: PhantomData::<O>.runtime(),
        }
    }

    fn fill<EG: Numeric, RV: ReadView<EG, Config = CmmaConfig>>(
        self_: &mut Self,
        read_view: &RV,
        #[comptime] ident: Ident,
        #[comptime] config: Self::Config,
    ) {
        RV::load_shared_memory::<ES, O>(read_view, &mut self_.smem, ident, config)
    }

    fn get_tile(
        self_: &Self,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: Self::Config,
    ) -> &Slice<'_, Line<ES>> {
        let stage_dim = config.stage_dim(ident);

        let nth_tile = O::to_nth_tile(x, y, stage_dim.num_tiles_x, stage_dim.num_tiles_y);

        let tile_stride = stage_dim.tile_num_elements() / config.line_size(ident);
        let start = nth_tile * tile_stride;

        self_.smem.slice(start, start + tile_stride)
    }
}
