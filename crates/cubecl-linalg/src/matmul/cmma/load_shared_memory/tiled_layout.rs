use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[derive(CubeType)]
pub(crate) struct TiledPosition {
    pub row: u32,
    pub col: u32,
}

#[cube]
pub(crate) trait TilingOrder {
    fn tile_row(nth_tile: u32, smem_tile_width: u32, smem_tile_height: u32) -> u32;
    fn tile_col(nth_tile: u32, smem_tile_width: u32, smem_tile_height: u32) -> u32;
}

pub(crate) struct RowMajorTiling {}
pub(crate) struct ColMajorTiling {}

#[cube]
impl TilingOrder for RowMajorTiling {
    fn tile_row(nth_tile: u32, smem_tile_width: u32, _smem_tile_height: u32) -> u32 {
        nth_tile / smem_tile_width
    }

    fn tile_col(nth_tile: u32, smem_tile_width: u32, _smem_tile_height: u32) -> u32 {
        nth_tile % smem_tile_width
    }
}

#[cube]
impl TilingOrder for ColMajorTiling {
    fn tile_row(nth_tile: u32, _smem_tile_width: u32, smem_tile_height: u32) -> u32 {
        nth_tile % smem_tile_height
    }

    fn tile_col(nth_tile: u32, _smem_tile_width: u32, smem_tile_height: u32) -> u32 {
        nth_tile / smem_tile_height
    }
}
