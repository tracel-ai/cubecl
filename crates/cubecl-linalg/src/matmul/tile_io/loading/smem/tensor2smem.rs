use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::cmma_matmul::{tile_num_elements, total_num_elements};
use crate::matmul::id_map::PlaneMapper;
use crate::matmul::tile_io::loading::smem::tiled_layout::{RowMajorTiling, TilingOrder};

#[cube]
pub trait Tensor2Smem {
    fn tensor_to_shared_memory<E: Numeric>(
        gmem: &Tensor<Line<E>>,
        smem: &mut SharedMemory<Line<E>>,
        gmem_row_offset: u32,
        gmem_col_offset: u32,
        #[comptime] block_info: BlockInfo,
    );
}

#[derive(CubeType)]
pub struct Tensor2SmemContinuous {}

#[cube]
impl PlaneMapper for Tensor2SmemContinuous {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }

    fn num_planes() -> u32 {
        CUBE_DIM_Y
    }

    fn plane_dim() -> u32 {
        CUBE_DIM_X
    }
}

#[cube]
impl Tensor2Smem for Tensor2SmemContinuous {
    fn tensor_to_shared_memory<E: Numeric>(
        gmem: &Tensor<Line<E>>,
        smem: &mut SharedMemory<Line<E>>,
        gmem_row_offset: u32,
        gmem_col_offset: u32,
        #[comptime] block_info: BlockInfo,
    ) {
        // TODO gives zeros for second column of RHS

        let num_smem_elements = comptime!(total_num_elements(block_info));
        let jump_length = comptime!(Self::num_planes() * gmem.line_size() * Self::plane_dim());

        let unit_position_base =
            (Self::plane_id() * Self::plane_dim() + Self::plane_unit()) * gmem.line_size();

        for i in 0..num_smem_elements / jump_length {
            let unit_position = unit_position_base + i * jump_length;

            let (row, col) = apply_tiled_layout(unit_position, block_info);

            load_single(
                gmem,
                smem,
                row + gmem_row_offset,
                col + gmem_col_offset,
                unit_position,
            )
        }
    }
}

#[cube]
pub(crate) fn apply_tiled_layout(
    unit_position: u32,
    #[comptime] block_info: BlockInfo,
) -> (u32, u32) {
    let tile_num_elements = tile_num_elements(block_info);
    let nth_tile = unit_position / tile_num_elements;
    let pos_within_tile = unit_position % tile_num_elements;

    // TODO allow col major too with generic. Should match with tile reader
    let (tile_row, tile_col) =
        RowMajorTiling::to_row_col(nth_tile, block_info.num_tiles_y, block_info.num_tiles_x);

    let row = tile_row * block_info.tile_size_x + pos_within_tile / block_info.tile_size_y;
    let col = tile_col * block_info.tile_size_y + pos_within_tile % block_info.tile_size_y;

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
    let read_pos = (read_row * gmem.stride(gmem.rank() - 2)
        + read_col * gmem.stride(gmem.rank() - 1))
        / gmem.line_size();

    smem[write_position] = Line::cast_from(gmem[read_pos]);
}
