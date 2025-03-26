use std::marker::PhantomData;

use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::stage::{StageConfig, TilingLayout};
use crate::matmul::components::tile::Tile;
use crate::matmul::components::{Ident, InputIdent, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct Stage<ES: Numeric, T: TilingLayout> {
    smem: SharedMemory<Line<ES>>,
    #[cube(comptime)]
    tiling_layout: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Stage<ES, T> {
    /// Instantiate a new stage for the given identifier
    pub fn new<S: StageConfig>(#[comptime] ident: Ident, #[comptime] config: S) -> Stage<ES, T> {
        let line_size = config.line_size(ident);

        let smem = SharedMemory::new_lined(
            comptime!(config.tiling_dimensions(ident).total_size() / line_size),
            line_size,
        );

        Self::new_with_smem(smem)
    }

    /// Instantiate a new stage for the given identifier
    pub fn new_aligned<S: StageConfig>(
        #[comptime] ident: Ident,
        #[comptime] alignment: u32,
        #[comptime] config: S,
    ) -> Stage<ES, T> {
        let line_size = config.line_size(ident);

        let smem = SharedMemory::new_aligned(
            comptime!(config.tiling_dimensions(ident).total_size() / line_size),
            line_size,
            alignment,
        );

        Self::new_with_smem(smem)
    }

    /// Instantiate with a custom shared memory
    pub fn new_with_smem(smem: SharedMemory<Line<ES>>) -> Stage<ES, T> {
        Stage::<ES, T> {
            smem,
            tiling_layout: PhantomData::<T>,
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
        T::get_tile::<ES, S>(self, x, y, ident, config)
    }

    /// Return the whole stage as a slice, for reading
    pub fn as_slice(&self) -> Slice<Line<ES>> {
        self.smem.to_slice()
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self) -> SliceMut<Line<ES>> {
        self.smem.to_slice_mut()
    }

    pub fn clear<S: StageConfig>(&mut self, #[comptime] ident: Ident, #[comptime] config: S) {
        // TODO: this assumes the stage was created with new
        let smem_length =
            comptime!(config.tiling_dimensions(ident).total_size() / config.line_size(ident));

        let unit_count = config.num_planes() * config.plane_dim();
        let num_writes_per_unit = smem_length.div_ceil(unit_count);

        let unit_base_position = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        for i in 0..num_writes_per_unit {
            let offset = unit_base_position + i * unit_count;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(smem_length % unit_count == 0) {
                self.smem[offset] = Line::cast_from(0);
            } else {
                if offset < smem_length {
                    self.smem[offset] = Line::cast_from(0);
                }
            }
        }
    }

    pub fn clear_buffer<S: StageConfig>(
        &mut self,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) {
        // // TODO: this assumes the stage was created with new
        // // Also assumes two buffers
        let tiling_dimensions = config.tiling_dimensions(ident);
        let line_size = config.line_size(ident);
        let smem_length = comptime!(tiling_dimensions.total_size() / line_size);
        let buffer_length = smem_length / 2;

        let matrix_layout = config.matrix_layout(ident);

        let unit_count = config.num_planes() * config.plane_dim();
        let num_writes_per_unit = buffer_length.div_ceil(unit_count);

        let unit_base_position = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        for i in 0..num_writes_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            let smem_position = match (ident.as_input(), matrix_layout) {
                (InputIdent::Lhs, MatrixLayout::ColMajor)
                | (InputIdent::Rhs, MatrixLayout::RowMajor) => {
                    buffer_id.to_u32() * buffer_length + unit_position
                }
                (InputIdent::Lhs, MatrixLayout::RowMajor) => {
                    let buffer_width = tiling_dimensions.tile_shape_col() / line_size;
                    buffer_id.to_u32() * buffer_width
                        + unit_position
                        + (unit_position / buffer_width) * buffer_width
                }
                (InputIdent::Rhs, MatrixLayout::ColMajor) => {
                    let buffer_height = tiling_dimensions.tile_shape_row() / line_size;
                    buffer_id.to_u32() * buffer_height
                        + unit_position
                        + (unit_position / buffer_height) * buffer_height
                }
            };

            #[allow(clippy::collapsible_else_if)]
            if comptime!(buffer_length % unit_count == 0) {
                self.smem[smem_position] = Line::cast_from(0);
            } else {
                if smem_position < smem_length {
                    self.smem[smem_position] = Line::cast_from(0);
                }
            }
        }
    }
}
