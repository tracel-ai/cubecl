use std::marker::PhantomData;

use crate::matmul::components::Ident;
use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::stage::{StageConfig, TilingLayout};
use crate::matmul::components::tile::Tile;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Each buffer has its own underlying shared memory
pub struct PhysicalDualStage<ES: Numeric, T: TilingLayout> {
    buffer_a: SharedMemory<Line<ES>>,
    buffer_b: SharedMemory<Line<ES>>,
    #[cube(comptime)]
    tiling_layout: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> PhysicalDualStage<ES, T> {
    pub fn new<S: StageConfig>(
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> PhysicalDualStage<ES, T> {
        let line_size = config.line_size(ident);

        let buffer_size = comptime!(config.tiling_dimensions(ident).total_size() / line_size);

        let buffer_a = SharedMemory::new_lined(buffer_size, line_size);
        let buffer_b = SharedMemory::new_lined(buffer_size, line_size);

        PhysicalDualStage::<ES, T> {
            buffer_a,
            buffer_b,
            tiling_layout: PhantomData,
        }
    }

    pub fn clear_buffer<S: StageConfig>(
        &mut self,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) {
        // TODO remove
    }

    pub fn as_slice(&self, #[comptime] buffer_id: BufferId) -> Slice<Line<ES>> {
        match buffer_id {
            BufferId::A => self.buffer_a.to_slice(),
            BufferId::B => self.buffer_b.to_slice(),
        }
    }

    pub fn as_slice_mut(&mut self, #[comptime] buffer_id: BufferId) -> SliceMut<Line<ES>> {
        match buffer_id {
            BufferId::A => self.buffer_a.to_slice_mut(),
            BufferId::B => self.buffer_b.to_slice_mut(),
        }
    }

    /// Get the tile at position (x,y) regardless of matrix layout
    pub fn get_tile<S: StageConfig>(
        &self,
        x_in_buffer: u32,
        y_in_buffer: u32,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        let tiling = config.tiling_dimensions(ident);

        let slice = match buffer_id {
            BufferId::A => &self.buffer_a.to_slice(),
            BufferId::B => &self.buffer_b.to_slice(),
        };

        T::get_tile::<ES, S>(
            slice,
            x_in_buffer,
            y_in_buffer,
            tiling.tile_count_row(),
            tiling.tile_count_col(),
            ident,
            config,
        )
    }
}
