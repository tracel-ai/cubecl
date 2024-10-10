use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::data::GlobalView;
use crate::matmul::id_map::PlaneMapper;
use crate::matmul::stage_info::{tile_num_elements, StageInfo};

use super::TensorView;

#[cube]
pub trait Smem2Tensor {
    fn smem_to_tensor<E: Numeric, C: CubePrimitive>(
        out: &mut TensorView<E>,
        smem_slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] stage_info: StageInfo,
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
        out: &mut TensorView<E>,
        tile_slice: &Slice<'_, C>,
        row_tile_begin: u32,
        col_tile_begin: u32,
        #[comptime] stage_info: StageInfo,
    ) {
        let unit_jump = Self::plane_dim() * out.tensor.line_size();
        let num_unit_writes = tile_num_elements(stage_info) / unit_jump;

        for i in 0..num_unit_writes {
            let unit_write = Self::plane_unit() * out.tensor.line_size() + i * unit_jump;

            let row = row_tile_begin + unit_write / stage_info.tile_size_y;
            let col = col_tile_begin + unit_write % stage_info.tile_size_y;

            let value = tile_slice[unit_write];
            TensorView::write_single::<C>(out, row, col, value);
        }
    }
}
