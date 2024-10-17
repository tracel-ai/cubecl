use super::base::Stage;
use super::TilingOrder;
use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::matmul_global::ReadView;
use crate::matmul::matrix_layout::{MatrixLayout, TensorIdent};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryStage<E: Numeric, O: TilingOrder> {
    smem: SharedMemory<Line<E>>,
    layout: MatrixLayout,
    line_size: u32,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<ES: Numeric, O: TilingOrder> Stage<ES> for SharedMemoryStage<ES, O> {
    type Underlying = SharedMemory<Line<ES>>;
    type Config = CmmaConfig;

    fn new(
        layout: MatrixLayout,
        #[comptime] line_size: u32,
        #[comptime] ident: TensorIdent,
        #[comptime] config: Self::Config,
    ) -> Self {
        let smem = SharedMemory::new_lined(
            comptime!(config.stage_num_elems(ident) / line_size),
            line_size,
        );

        SharedMemoryStage::<ES, O> {
            smem,
            layout,
            line_size,
            _tiling_order: PhantomData::<O>.runtime(),
        }
    }

    fn fill<EG: Numeric, RV: ReadView<EG, Config = CmmaConfig>>(
        self_: &mut Self,
        read_view: &RV,
        #[comptime] ident: TensorIdent,
        #[comptime] config: Self::Config,
    ) {
        RV::load_shared_memory::<ES, O>(read_view, &mut self_.smem, ident, config)
    }

    fn get_tile(
        self_: &Self,
        x: u32,
        y: u32,
        #[comptime] ident: TensorIdent,
        #[comptime] config: Self::Config,
    ) -> (&Slice<'_, Line<ES>>, MatrixLayout) {
        let stage_dim = config.stage_dim(ident);

        let nth_tile = O::to_nth_tile(x, y, stage_dim.num_tiles_x, stage_dim.num_tiles_y);

        let tile_stride = stage_dim.tile_num_elements() / self_.line_size;
        let start = nth_tile * tile_stride;

        (self_.smem.slice(start, start + tile_stride), self_.layout)
    }

    fn layout(self_: &Self) -> MatrixLayout {
        self_.layout
    }
}
