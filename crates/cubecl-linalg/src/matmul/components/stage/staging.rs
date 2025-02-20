use std::marker::PhantomData;

use crate::matmul::components::stage::{StageConfig, TilingLayoutTrait};
use crate::matmul::components::tile::Tile;
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct Stage<ES: Numeric, T: TilingLayoutTrait> {
    pub smem: SharedMemory<Line<ES>>,
    tiling_layout: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayoutTrait> Stage<ES, T> {
    /// Instantiate a new stage for the given identifier
    pub fn new<S: StageConfig>(#[comptime] ident: Ident, #[comptime] config: S) -> Stage<ES, T> {
        let line_size = config.line_size(ident);

        let smem = SharedMemory::new_lined(
            comptime!(config.tiling_dimensions(ident).total_size() / line_size),
            line_size,
        );

        Stage::<ES, T> {
            smem,
            tiling_layout: PhantomData::<T>.runtime(),
        }
    }

    /// Get the tile at position (x,y) regardless of matrix layout
    pub fn get_tile<S: StageConfig>(
        &self,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        T::get_tile::<ES, S>(self.smem.to_slice(), x, y, ident, config)
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self) -> SliceMut<Line<ES>> {
        self.smem.to_slice_mut()
    }
}
