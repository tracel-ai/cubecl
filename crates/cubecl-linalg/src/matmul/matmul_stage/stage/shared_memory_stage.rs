use super::base::Stage;
use super::TilingOrder;
use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_tile::new_ref_tile;
use crate::matmul::matmul_tile::RefTile;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use crate::matmul::stage_info::{tile_num_elements, total_num_elements};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryStage<E: Numeric, O: TilingOrder> {
    smem: SharedMemory<Line<E>>,
    layout: MatrixLayout,
    stage_info: StageInfo,
    line_size: u32,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<E: Numeric, O: TilingOrder> Stage<E> for SharedMemoryStage<E, O> {
    type Underlying = SharedMemory<Line<E>>;

    fn new(
        layout: MatrixLayout,
        #[comptime] stage_info: StageInfo,
        #[comptime] line_size: u32,
    ) -> Self {
        let smem = SharedMemory::new_lined(
            comptime!(total_num_elements(stage_info) / line_size),
            line_size,
        );

        SharedMemoryStage::<E, O> {
            smem,
            layout,
            stage_info: stage_info.runtime(),
            line_size,
            _tiling_order: PhantomData::<O>.runtime(),
        }
    }

    fn fill<EG: Numeric, G: GlobalView<EG>>(stage: &mut Self, gmem: &G) {
        G::load_shared_memory::<E, O>(gmem, &mut stage.smem, stage.stage_info)
    }

    fn get_tile(stage: &Self, x: u32, y: u32) -> RefTile<'_, E> {
        let nth_tile = O::to_nth_tile(
            x,
            y,
            stage.stage_info.num_tiles_x,
            stage.stage_info.num_tiles_y,
        );

        let tile_stride = tile_num_elements(stage.stage_info) / stage.line_size;
        let start = nth_tile * tile_stride;

        new_ref_tile(
            x,
            y,
            stage.smem.slice(start, start + tile_stride),
            stage.layout,
        )
    }

    fn layout(stage: &Self) -> MatrixLayout {
        stage.layout
    }
}
