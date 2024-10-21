use crate::matmul::cmma_matmul::stage::{TilingOrder, XMajorTiling, YMajorTiling};
use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::TilingOrderConfig;

#[derive(CubeType, Clone, Copy)]
pub struct Stage<ES: Numeric> {
    pub smem: SharedMemory<Line<ES>>,
}

#[cube]
impl<ES: Numeric> Stage<ES> {
    pub fn new<S: SmmConfig>(#[comptime] ident: Ident, #[comptime] config: S) -> Stage<ES> {
        let smem = SharedMemory::new_lined(
            comptime!(config.stage_dim(ident).num_elements() / config.line_size(ident)),
            config.line_size(ident),
        );

        Stage::<ES> { smem }
    }

    pub fn get_tile<S: SmmConfig>(
        &self,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> &Slice<'_, Line<ES>> {
        let stage_dim = config.stage_dim(ident);

        let nth_tile = match config.tiling_order() {
            TilingOrderConfig::XMajor => {
                XMajorTiling::to_nth_tile(x, y, stage_dim.num_tiles_x, stage_dim.num_tiles_y)
            }
            TilingOrderConfig::YMajor => {
                YMajorTiling::to_nth_tile(x, y, stage_dim.num_tiles_x, stage_dim.num_tiles_y)
            }
        };

        let tile_stride = stage_dim.tile_num_elements() / config.line_size(ident);
        let start = nth_tile * tile_stride;

        self.smem.slice(start, start + tile_stride)
    }

    pub fn as_slice_mut(&mut self) -> &mut SliceMut<'_, Line<ES>> {
        self.smem.as_slice_mut()
    }
}
