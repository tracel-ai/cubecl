use crate::matmul::components::stage::{StageConfig, TilingLayout};
use crate::matmul::components::tile::Tile;
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
    pub fn new<S: StageConfig>(#[comptime] ident: Ident, #[comptime] config: S) -> Stage<ES> {
        let line_size = config.line_size(ident);

        let smem = SharedMemory::new_lined(
            comptime!(config.tiling(ident).total_size() / line_size),
            line_size,
        );

        Stage::<ES> { smem }
    }

    /// Get the tile at position (x,y) regardless of layout, as a contiguous slice
    pub fn get_tile<S: StageConfig>(
        &self,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        let tiling = config.tiling(ident);

        let nth_tile = TilingLayout::to_nth_tile(
            config.tiling_layout(ident),
            x,
            y,
            tiling.tile_count_row(),
            tiling.tile_count_col(),
        );

        let tile_stride = tiling.tile_size() / config.line_size(ident);
        let start = nth_tile * tile_stride;

        Tile::new_contiguous::<S::TmmConfig>(
            self.smem.slice(start, start + tile_stride),
            ident,
            config.to_tmm_config(),
        )
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self) -> SliceMut<Line<ES>> {
        self.smem.to_slice_mut()
    }
}
