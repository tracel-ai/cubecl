use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::{BlockInfo, tile_num_elements};
use crate::matmul::id_map::PlaneMapper;

#[cube]
pub trait Smem2Tensor {
    fn smem_to_tensor<E: Numeric, C: CubePrimitive>(
        out: &mut Tensor<Line<E>>,
        smem_slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        cube_offsets: (u32, u32),
        #[comptime] block_info: BlockInfo,
    );
}

#[derive(CubeType)]
pub struct Smem2TensorSimple {}

#[cube]
impl PlaneMapper for Smem2TensorSimple {
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
impl Smem2Tensor for Smem2TensorSimple {
    fn smem_to_tensor<E: Numeric, C: CubePrimitive>(
        gmem: &mut Tensor<Line<E>>,
        smem_slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        cube_offsets: (u32, u32),
        #[comptime] block_info: BlockInfo,
    ) {
        let row_tile_begin = (cube_offsets.0 + compute_plane_offset) * block_info.tile_size_x;
        let col_tile_begin = (cube_offsets.1 + accumulator_offset) * block_info.tile_size_y;

        let unit_jump = Self::plane_dim() * gmem.line_size();
        let num_unit_writes = tile_num_elements(block_info) / unit_jump;

        for i in 0..num_unit_writes {
            let unit_write = Self::plane_unit() * gmem.line_size() + i * unit_jump;

            let row = row_tile_begin + unit_write / block_info.tile_size_y;
            let col = col_tile_begin + unit_write % block_info.tile_size_y;

            write_single(gmem, smem_slice, unit_write, row, col);
        }
    }
}

#[cube]
/// Assumes (write_row, write_col) is within bounds
/// Does not account for batch offset
fn write_single<E: Numeric, C: CubePrimitive>(
    gmem: &mut Tensor<Line<E>>,
    smem_slice: &Slice<'_, C>,
    read_position: u32,
    write_row: u32,
    write_col: u32,
) {
    let write_position = (write_row * gmem.stride(gmem.rank() - 2)
        + write_col * gmem.stride(gmem.rank() - 1))
        / gmem.line_size();
    gmem[write_position] = Line::cast_from(smem_slice[read_position]);
}
