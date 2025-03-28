use std::marker::PhantomData;

use crate::matmul::components::global::multi_stage::double_buffering::{BufferId, BufferIdExpand};
use crate::matmul::components::stage::{StageConfig, TilingLayout};
use crate::matmul::components::tile::Tile;
use crate::matmul::components::{Ident, InputIdent, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Determines which [DualStage] to use
#[derive(CubeType, Clone, Copy)]
pub enum DualStageFormat {
    Virtual,
    Physical,
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for double buffering global matmuls
pub enum DualStage<ES: Numeric, T: TilingLayout> {
    Virtual(VirtualDualStage<ES, T>),
    Physical(PhysicalDualStage<ES, T>),
}

#[cube]
impl<ES: Numeric, T: TilingLayout> DualStage<ES, T> {
    pub fn new<S: StageConfig>(
        dual_stage_format: DualStageFormat,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Self {
        match dual_stage_format {
            DualStageFormat::Virtual => {
                DualStage::new_Virtual(VirtualDualStage::new::<S>(ident, config))
            }
            DualStageFormat::Physical => {
                DualStage::new_Physical(PhysicalDualStage::new::<S>(ident, config))
            }
        }
    }

    pub fn clear_buffer<S: StageConfig>(
        &mut self,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) {
        match self {
            DualStage::Virtual(virtual_dual_stage) => {
                virtual_dual_stage.clear_buffer::<S>(buffer_id, ident, config)
            }
            DualStage::Physical(physical_dual_stage) => {
                physical_dual_stage.clear_buffer::<S>(buffer_id, ident, config)
            }
        }
    }

    /// Get the tile at position (x,y) regardless of matrix layout
    pub fn get_tile<S: StageConfig>(
        &self,
        nth_tile_in_buffer: u32,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        match self {
            DualStage::Virtual(virtual_dual_stage) => {
                virtual_dual_stage.get_tile::<S>(nth_tile_in_buffer, buffer_id, ident, config)
            }
            DualStage::Physical(physical_dual_stage) => {
                physical_dual_stage.get_tile::<S>(nth_tile_in_buffer, buffer_id, ident, config)
            }
        }
    }
}

#[derive(CubeType, Clone, Copy)]
/// There is only one underlying shared memory, buffers are split with index calculations
pub struct VirtualDualStage<ES: Numeric, T: TilingLayout> {
    smem: SharedMemory<Line<ES>>,
    #[cube(comptime)]
    tiling_layout: PhantomData<T>,
}

#[derive(CubeType, Clone, Copy)]
/// Each buffer has its own underlying shared memory
pub struct PhysicalDualStage<ES: Numeric, T: TilingLayout> {
    buffer_a: SharedMemory<Line<ES>>,
    buffer_b: SharedMemory<Line<ES>>,
    #[cube(comptime)]
    tiling_layout: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> VirtualDualStage<ES, T> {
    fn new<S: StageConfig>(
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> VirtualDualStage<ES, T> {
        let line_size = config.line_size(ident);

        let smem = SharedMemory::new_lined(
            comptime!(config.tiling_dimensions(ident).total_size() / line_size),
            line_size,
        );

        VirtualDualStage::<ES, T> {
            smem,
            tiling_layout: PhantomData,
        }
    }

    fn clear_buffer<S: StageConfig>(
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

    pub fn as_slice(&self) -> Slice<Line<ES>> {
        self.smem.to_slice()
    }

    pub fn as_slice_mut(&mut self) -> SliceMut<Line<ES>> {
        self.smem.to_slice_mut()
    }

    /// Get the tile at position (x,y) regardless of matrix layout
    pub fn get_tile<S: StageConfig>(
        &self,
        nth_tile_in_buffer: u32,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        let (x, y) = match ident.as_input() {
            InputIdent::Lhs => (nth_tile_in_buffer, buffer_id.to_u32().runtime()),
            InputIdent::Rhs => (buffer_id.to_u32().runtime(), nth_tile_in_buffer),
        };

        T::get_tile::<ES, S>(&self.smem.to_slice(), x, y, ident, config)
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> PhysicalDualStage<ES, T> {
    fn new<S: StageConfig>(
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> PhysicalDualStage<ES, T> {
        let line_size = config.line_size(ident);
        let buffer_size = comptime!(config.tiling_dimensions(ident).total_size() / (2 * line_size));

        let buffer_a = SharedMemory::new_lined(buffer_size, line_size);
        let buffer_b = SharedMemory::new_lined(buffer_size, line_size);

        PhysicalDualStage::<ES, T> {
            buffer_a,
            buffer_b,
            tiling_layout: PhantomData,
        }
    }

    fn clear_buffer<S: StageConfig>(
        &mut self,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) {
        // TODO
        // Should be similar to clear stage in mono
    }

    pub fn as_slice(&self, buffer_id: BufferId) -> Slice<Line<ES>> {
        match buffer_id {
            BufferId::A => self.buffer_a.to_slice(),
            BufferId::B => self.buffer_b.to_slice(),
        }
    }

    pub fn as_slice_mut(&mut self, buffer_id: BufferId) -> SliceMut<Line<ES>> {
        match buffer_id {
            BufferId::A => self.buffer_a.to_slice_mut(),
            BufferId::B => self.buffer_b.to_slice_mut(),
        }
    }

    /// Get the tile at position (x,y) regardless of matrix layout
    pub fn get_tile<S: StageConfig>(
        &self,
        nth_tile_in_buffer: u32,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        let (x, y) = match ident.as_input() {
            InputIdent::Lhs => (nth_tile_in_buffer, 0u32.runtime()),
            InputIdent::Rhs => (0u32.runtime(), nth_tile_in_buffer),
        };

        match buffer_id {
            BufferId::A => T::get_tile::<ES, S>(&self.buffer_a.to_slice(), x, y, ident, config),
            BufferId::B => T::get_tile::<ES, S>(&self.buffer_b.to_slice(), x, y, ident, config),
        }
    }
}
