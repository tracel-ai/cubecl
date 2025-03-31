use std::marker::PhantomData;

use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::stage::{StageConfig, TilingLayout};
use crate::matmul::components::tile::Tile;
use crate::matmul::components::{Ident, InputIdent};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// There is only one underlying shared memory, buffers are split with index calculations
pub struct VirtualDualStage<ES: Numeric, T: TilingLayout> {
    smem: SharedMemory<Line<ES>>,
    #[cube(comptime)]
    tiling_layout: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> VirtualDualStage<ES, T> {
    pub fn new<S: StageConfig>(
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> VirtualDualStage<ES, T> {
        let line_size = config.line_size(ident);
        let num_buffers = 2;

        let smem = SharedMemory::new_lined(
            comptime!(config.tiling_dimensions(ident).total_size() * num_buffers / line_size),
            line_size,
        );

        VirtualDualStage::<ES, T> {
            smem,
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

    pub fn as_slice(&self) -> Slice<Line<ES>> {
        self.smem.to_slice()
    }

    pub fn as_slice_mut(&mut self) -> SliceMut<Line<ES>> {
        self.smem.to_slice_mut()
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

        let x = x_in_buffer + tiling.tile_count_row() * buffer_id.to_u32();
        let y = y_in_buffer + tiling.tile_count_col() * buffer_id.to_u32();

        let (double_buffered_tile_count_row, double_buffered_tile_count_col) =
            match ident.as_input() {
                InputIdent::Lhs => (tiling.tile_count_row() * 2, tiling.tile_count_col()),
                InputIdent::Rhs => (tiling.tile_count_row(), tiling.tile_count_col() * 2),
            };

        T::get_tile::<ES, S>(
            &self.smem.to_slice(),
            x,
            y,
            double_buffered_tile_count_row,
            double_buffered_tile_count_col,
            ident,
            config,
        )
    }
}
