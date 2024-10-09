use super::Stage;
use crate::matmul::stage_info::{tile_num_elements, total_num_elements};
use crate::matmul::data::{ArrayView, Tile};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::array_to_shared_memory;
use crate::matmul::tile_io::loading::tiled_layout::{RowMajorTiling, TilingOrder};
use crate::matmul::{stage_info::StageInfo, data::new_tile};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct ArrayStage<E: Numeric> {
    smem: SharedMemory<Line<E>>,
    layout: MatrixLayout,
    block_info: StageInfo,
}

#[cube]
impl<E: Numeric> Stage<E> for ArrayStage<E> {
    type GlobalView = ArrayView<E>;
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

        ArrayStage::<E> {
            smem,
            layout,
            block_info: block_info.runtime(),
        }
    }

    fn fill(block: &mut Self, gmem: &Self::GlobalView) {
        // TODO we don't want unit_pos_y here
        array_to_shared_memory::<E, E>(&gmem.array, &mut block.smem, UNIT_POS_Y, block.block_info)
    }

    fn get_tile(block: &Self, x: u32, y: u32) -> Tile<'_, E> {
        let tile_stride = tile_num_elements(block.block_info);

        let nth_tile = RowMajorTiling::to_nth_tile(
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
