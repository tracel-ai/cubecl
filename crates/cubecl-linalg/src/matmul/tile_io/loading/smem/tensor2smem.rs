use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::num_elements;
use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::cmma_old::load_shared_memory::tiled_layout::RowMajorTiling;
use crate::matmul::cmma_old::load_shared_memory::tiled_layout::TilingOrder;

#[cube]
pub(crate) fn tensor_to_shared_memory<E: Numeric>(
    gmem: &Tensor<Line<E>>,
    smem: &mut SharedMemory<Line<E>>,
    gmem_row_offset: u32,
    gmem_col_offset: u32,
    #[comptime] block_info: BlockInfo,
) {
    // TODO generalize
    let plane_dim = 32;
    let num_load_planes = 1;

    let num_smem_elements = num_elements(block_info);
    for i in 0..num_smem_elements {
        smem[i] = Line::cast_from(0);
    }

    let line_size = gmem.line_size();
    let jump_length = num_load_planes * line_size * plane_dim;
    let num_iterations = num_smem_elements / jump_length;

    // TODO id system
    let lane_id = UNIT_POS_X;
    let plane_id = UNIT_POS_Y;
    let unit_position_base = (plane_id * plane_dim + lane_id) * line_size;

    // #[unroll]
    for i in 0..num_iterations {
        let unit_position = unit_position_base + i * jump_length;
        let write_position = unit_position;

        let (row, col) = apply_tiled_layout(unit_position, block_info);
        let gmem_row = row + gmem_row_offset;
        let gmem_col = col + gmem_col_offset;

        load_single(gmem, smem, gmem_row, gmem_col, write_position)
    }
}

#[cube]
pub(crate) fn apply_tiled_layout(
    unit_position: u32,
    #[comptime] block_info: BlockInfo,
) -> (u32, u32) {
    let num_tile_elements = num_elements(block_info);
    let smem_tile_width = block_info.num_tiles_y;
    let smem_tile_height = block_info.num_tiles_x;

    let nth_tile = unit_position / num_tile_elements;

    // TODO allow col major too with generic. Should match with tile reader
    let (tile_row, tile_col) =
        RowMajorTiling::to_row_col(nth_tile, smem_tile_width, smem_tile_height);

    let tile_stride = block_info.tile_size_y;
    let pos_within_tile = unit_position % num_tile_elements;
    let row_within_tile = pos_within_tile / tile_stride;
    let col_within_tile = pos_within_tile % tile_stride;

    let row = tile_row * block_info.tile_size_x + row_within_tile;
    let col = tile_col * block_info.tile_size_y + col_within_tile;

    (row, col)
}

#[cube]
/// Loads one line from gmem at position (read_row, read_col)
/// and writes it as casted in smem at position write_position
///
/// Assumes (read_row, read_col) is within bounds
/// Does not account for batch offset
fn load_single<EG: Numeric, ES: Numeric>(
    gmem: &Tensor<Line<EG>>,
    smem: &mut SharedMemory<Line<ES>>,
    read_row: u32,
    read_col: u32,
    write_position: u32,
) {
    let line_size = gmem.line_size();

    let read_pos = (read_row * gmem.stride(gmem.rank() - 2)
        + read_col * gmem.stride(gmem.rank() - 1))
        / line_size;

    smem[write_position] = Line::cast_from(gmem[read_pos]);
}
