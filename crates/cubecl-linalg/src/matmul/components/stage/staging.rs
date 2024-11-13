use crate::matmul::components::stage::tiling_order::{
    TilingOrderConfig, XMajorTiling, YMajorTiling,
};
use crate::matmul::components::stage::{Config, TilingOrder};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct Stage<ES: Numeric> {
    pub smem: SharedMemory<Line<ES>>,
}

#[cube]
impl<ES: Numeric> Stage<ES> {
    /// Instantiate a new stage for the given identifier
    pub fn new<S: Config>(#[comptime] ident: Ident, #[comptime] config: S) -> Stage<ES> {
        let line_size = config.line_size(ident);

        let smem = SharedMemory::new_lined(
            comptime!(config.stage_dim(ident).num_elements() / line_size),
            line_size,
        );

        Stage::<ES> { smem }
    }

    /// Get the tile at position (x,y) regardless of layout, as a contiguous slice
    pub fn get_tile<S: Config>(
        &self,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Slice<Line<ES>> {
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

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self) -> SliceMut<Line<ES>> {
        self.smem.to_slice_mut()
    }
}
