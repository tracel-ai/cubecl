use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::config::{MatmulConfig, PlaneMapper};
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matmul_stage::TilingOrder;
use crate::matmul::stage_info::{tile_num_elements, total_num_elements, StageInfo};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::ReadView;

#[cube]
pub trait SharedMemoryLoader: Clone + Copy + Send + Sync + 'static {
    type Config: GmmConfig;

    fn load_shared_memory<EG: Numeric, ES: Numeric, RV: ReadView<EG>, O: TilingOrder>(
        gmem: &RV,
        smem: &mut SharedMemory<Line<ES>>,
        #[comptime] line_size: u32,
        #[comptime] stage_info: StageInfo,
        #[comptime] config: Self::Config,
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
}

#[cube]
impl SharedMemoryLoader for Gmem2SmemContinuous {
    type Config = CmmaConfig;

    fn load_shared_memory<EG: Numeric, ES: Numeric, RV: ReadView<EG>, O: TilingOrder>(
        gmem: &RV,
        smem: &mut SharedMemory<Line<ES>>,
        #[comptime] line_size: u32,
        #[comptime] stage_info: StageInfo,
        #[comptime] config: Self::Config,
    ) {
        let num_stage_elements = comptime!(total_num_elements(stage_info));

        let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
        let _ = comptime!(check_jump_divides_well(num_stage_elements, jump_length));

        let unit_position_base =
            (Self::plane_id() * config.plane_dim() + Self::plane_unit()) * line_size;

        for i in 0..num_stage_elements / jump_length {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = tile_num_elements(stage_info);
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) =
                O::to_x_y(nth_tile, stage_info.num_tiles_x, stage_info.num_tiles_y);

            let line = RV::load_coalesced(
                gmem,
                tile_x,
                tile_y,
                pos_within_tile,
                stage_info.tile_size_x,
                stage_info.tile_size_y,
            );

            smem[unit_position / line_size] = Line::cast_from(line);
        }
    }
}

fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
        Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
