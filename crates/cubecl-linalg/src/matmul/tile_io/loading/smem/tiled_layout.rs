use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
/// Its role is to ensure consistency between tensor loader and tile reader
pub trait TilingOrder: Clone + 'static + Send + Sync {
    fn to_row_col(nth_tile: u32, smem_tile_width: u32, smem_tile_height: u32) -> (u32, u32);
    fn to_nth_tile(
        tile_row: u32,
        tile_col: u32,
        smem_tile_width: u32,
        smem_tile_height: u32,
    ) -> u32;
}

#[derive(Clone)]
pub struct RowMajorTiling {}
#[derive(Clone)]
pub struct ColMajorTiling {}

#[cube]
impl TilingOrder for RowMajorTiling {
    fn to_row_col(nth_tile: u32, smem_tile_width: u32, _smem_tile_height: u32) -> (u32, u32) {
        (nth_tile / smem_tile_width, nth_tile % smem_tile_width)
    }

    fn to_nth_tile(
        tile_row: u32,
        tile_col: u32,
        smem_tile_width: u32,
        _smem_tile_height: u32,
    ) -> u32 {
        tile_row * smem_tile_width + tile_col
    }
}

#[cube]
impl TilingOrder for ColMajorTiling {
    fn to_row_col(nth_tile: u32, _smem_tile_width: u32, smem_tile_height: u32) -> (u32, u32) {
        (nth_tile % smem_tile_height, nth_tile / smem_tile_height)
    }

    fn to_nth_tile(
        tile_row: u32,
        tile_col: u32,
        _smem_tile_width: u32,
        smem_tile_height: u32,
    ) -> u32 {
        tile_col * smem_tile_height + tile_row
    }
}
