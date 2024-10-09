use super::base::Stage;
use crate::matmul::data::{GlobalView, Tile};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::{tile_num_elements, total_num_elements};
use crate::matmul::tile_io::loading::tiled_layout::TilingOrder;
use crate::matmul::{data::new_tile, stage_info::StageInfo};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryStage<E: Numeric, O: TilingOrder> {
    smem: SharedMemory<Line<E>>,
    layout: MatrixLayout,
    block_info: StageInfo,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<E: Numeric, O: TilingOrder> Stage<E> for SharedMemoryStage<E, O> {
    type Underlying = SharedMemory<Line<E>>;

    fn new(
        layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
        #[comptime] line_size: u32,
    ) -> Self {
        let smem = SharedMemory::new_lined(
            comptime!(total_num_elements(block_info) / line_size),
            line_size,
        );

        SharedMemoryStage::<E, O> {
            smem,
            layout,
            block_info: block_info.runtime(),
            _tiling_order: PhantomData::<O>.runtime(),
        }
    }

    fn fill<EG: Numeric, G: GlobalView<EG>>(block: &mut Self, gmem: &G) {
        G::load_shared_memory::<E>(gmem, &mut block.smem, block.block_info)
    }

    fn get_tile(block: &Self, x: u32, y: u32) -> Tile<'_, E> {
        let tile_stride = tile_num_elements(block.block_info);

        let nth_tile = O::to_nth_tile(
            x,
            y,
            block.block_info.num_tiles_x,
            block.block_info.num_tiles_y,
        );

        let start = nth_tile * tile_stride;
        new_tile(
            x,
            y,
            block.smem.slice(start, start + tile_stride),
            block.layout,
        )
    }

    fn layout(block: &Self) -> MatrixLayout {
        block.layout
    }
}
