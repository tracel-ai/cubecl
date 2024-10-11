use crate::matmul::id_map::PlaneMapper;
use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_stage::TilingOrder;
use crate::matmul::stage_info::{tile_num_elements, total_num_elements, StageInfo};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait SharedMemoryLoader: Clone + Copy + Send + Sync + 'static {
    fn load_shared_memory<EG: Numeric, ES: Numeric, G: GlobalView<EG>, O: TilingOrder>(
        gmem: &G,
        smem: &mut SharedMemory<Line<ES>>,
        #[comptime] block_info: StageInfo,
    );
}

#[derive(CubeType, Clone, Copy)]
pub struct Gmem2SmemContinuous {}

#[cube]
impl PlaneMapper for Gmem2SmemContinuous {
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
impl SharedMemoryLoader for Gmem2SmemContinuous {
    fn load_shared_memory<EG: Numeric, ES: Numeric, G: GlobalView<EG>, O: TilingOrder>(
        gmem: &G,
        smem: &mut SharedMemory<Line<ES>>,
        #[comptime] stage_info: StageInfo,
    ) {
        let line_size = G::line_size(gmem);
        let num_smem_elements = comptime!(total_num_elements(stage_info));

        // Could be comptime if we were able to fetch line_size as comptime
        let jump_length = Self::num_planes() * line_size * Self::plane_dim();

        let unit_position_base =
            (Self::plane_id() * Self::plane_dim() + Self::plane_unit()) * line_size;

        for i in 0..num_smem_elements / jump_length {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = tile_num_elements(stage_info);
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) =
                O::to_x_y(nth_tile, stage_info.num_tiles_x, stage_info.num_tiles_y);

            let line = G::load_coalesced(
                gmem,
                tile_x,
                tile_y,
                pos_within_tile,
                stage_info.tile_size_x,
                stage_info.tile_size_y,
            );
            smem[unit_position] = Line::cast_from(line);
        }
    }
}

// #[cube]
// pub(crate) fn tile_row_col<T: TilingOrder>(
//     unit_position: u32,
//     #[comptime] stage_info: StageInfo,
// ) -> (u32, u32) {
//     let row = tile_row * stage_info.tile_size_x + pos_within_tile / stage_info.tile_size_y;
//     let col = tile_col * stage_info.tile_size_y + pos_within_tile % stage_info.tile_size_y;

//     (row, col)
// }
