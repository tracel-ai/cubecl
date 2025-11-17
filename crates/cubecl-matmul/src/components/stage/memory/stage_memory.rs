use std::marker::PhantomData;

use crate::components::global::{
    GlobalReaderConfig, RoleRule, RoleRuleConfig, SpecializedLoadingSides,
};
use crate::components::stage::{StageMemoryConfig, TilingLayout};
use crate::components::tile::StridedTile;
use crate::components::{MatmulIdent, MatrixLayout, StageIdent};
use crate::components::{global::read::StageBuffer, stage::StageFamily};
use crate::components::{stage::Stage, tile::io::Strided};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

pub struct StridedStageFamily;

impl StageFamily for StridedStageFamily {
    type TileKind = Strided;

    type Stage<ES: Numeric, T: TilingLayout> = StridedStage<ES, T>;
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct StridedStage<ES: Numeric, T: TilingLayout> {
    /// Underlying shared memory
    pub smem: SharedMemory<Line<ES>>,
    buffer_index: u32,

    #[cube(comptime)]
    config: StageMemoryConfig,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StridedStage<ES, T> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(#[comptime] config: StageMemoryConfig) -> StridedStage<ES, T> {
        let line_size = config.line_size;

        let smem = SharedMemory::new_lined(
            comptime!(config.num_stages * config.elements_in_stage() / line_size),
            line_size,
        );

        Self::new_with_smem(smem, config)
    }

    /// Instantiate a new stage memory for the given identifier, with shared memory alignment
    pub fn new_aligned(
        #[comptime] alignment: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStage<ES, T> {
        let line_size = config.line_size;

        let smem = SharedMemory::new_aligned(
            comptime!(config.num_stages * config.elements_in_stage() / line_size),
            line_size,
            alignment,
        );

        Self::new_with_smem(smem, config)
    }

    /// Instantiate with a custom shared memory
    pub fn new_with_smem(
        smem: SharedMemory<Line<ES>>,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStage<ES, T> {
        StridedStage::<ES, T> {
            smem,
            config,
            buffer_index: 0u32,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn with_buffer_index(&self, buffer_idx: u32) -> Self {
        StridedStage::<ES, T> {
            smem: self.smem,
            config: self.config,
            buffer_index: buffer_idx,
            _phantom: PhantomData::<T>,
        }
    }

    /// Get the tile at position (row, col)
    pub fn get_tile(&self, tile: Coords2d) -> StridedTile<ES> {
        T::get_tile::<ES>(self, tile, self.config)
    }

    /// Get the tile at position (row, col)
    pub fn get_tile_mut(&self, tile: Coords2d) -> StridedTile<ES, ReadWrite> {
        let tile = self.get_tile(tile);
        StridedTile::<ES, ReadWrite> {
            slice: tile.slice.as_mut_unchecked(),
            stride: tile.stride,
            layout: tile.layout,
            line_size: tile.line_size,
        }
    }

    /// Return the whole stage as a slice, for reading
    pub fn as_slice(&self, #[comptime] line_size: u32) -> Slice<Line<ES>> {
        let stage_size = comptime![self.config.elements_in_stage() / self.smem.line_size()];
        let stage_offset = self.buffer_index * stage_size;
        self.smem
            .slice(stage_offset, stage_offset + stage_size)
            .with_line_size(line_size)
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self, #[comptime] line_size: u32) -> SliceMut<Line<ES>> {
        let stage_size = comptime![self.config.elements_in_stage() / self.smem.line_size()];
        let stage_offset = self.buffer_index * stage_size;
        self.smem
            .slice_mut(stage_offset, stage_offset + stage_size)
            .with_line_size(line_size)
    }

    /// Zero out the shared memory
    /// Available for matmul only
    pub fn clear_all(&mut self, #[comptime] config: GlobalReaderConfig) {
        // TODO: this assumes the stage was created with new
        let smem_length = comptime!(
            self.config.num_stages * self.config.elements_in_stage() / self.config.line_size
        );

        let unit_count = config.loading_units_count();
        let num_writes_per_unit = smem_length.div_ceil(unit_count);

        let unit_base_position = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;

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

    /// Zero out the shared memory for only one stage
    /// Available for matmul only
    pub fn clear_stage(
        &mut self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: GlobalReaderConfig,
        // #[comptime] ident: MatmulIdent,
        // #[comptime] matrix_layout: MatrixLayout,
        // #[comptime] num_loading_planes: u32,
        // #[comptime] plane_dim: u32,
        // #[comptime] role_rule_config: RoleRuleConfig,
        // #[comptime] specialized_loading_sides: SpecializedLoadingSides,
        // #[comptime] elements_in_tile_k: u32,
    ) {
        // TODO: this assumes the stage was created with new
        // Also assumes two buffers
        // let tiling_scheme = config.tiling_scheme();
        let line_size = comptime![self.config.line_size];
        let smem_length =
            comptime!(self.config.num_stages * self.config.elements_in_stage() / line_size);
        let buffer_length = smem_length / 2;

        let unit_count = config.loading_units_count();
        let num_writes_per_unit = buffer_length.div_ceil(unit_count);

        let unit_base_position = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;

        for i in 0..num_writes_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            let smem_position = match (config.stage_ident, config.smem_config.matrix_layout)
            {
                (StageIdent::Lhs, MatrixLayout::ColMajor)
                | (StageIdent::Rhs, MatrixLayout::RowMajor) => {
                    stage_buffer.to_index() * buffer_length + unit_position
                }
                (StageIdent::Lhs, MatrixLayout::RowMajor)
                | (StageIdent::Rhs, MatrixLayout::ColMajor) => {
                    let buffer_width = config.smem_config.elements_in_tile_contiguous_dim()
                        / config.smem_config.line_size;
                    stage_buffer.to_index() * buffer_width
                        + unit_position
                        + (unit_position / buffer_width) * buffer_width
                }
                _ => comptime!(unreachable!()),
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

    /// Frees the shared memory for reuse, if possible on the target runtime.
    ///
    /// # Safety
    /// *Must* be used in uniform control flow
    /// *Must not* have any dangling references to this shared memory
    pub unsafe fn free(self) {
        unsafe { self.smem.free() };
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Stage<ES, ReadOnly> for StridedStage<ES, T> {
    type TileKind = Strided;

    fn tile(this: &Self, tile: Coords2d) -> StridedTile<ES> {
        this.get_tile(tile)
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Stage<ES, ReadWrite> for StridedStage<ES, T> {
    type TileKind = Strided;

    fn tile(this: &Self, tile: Coords2d) -> StridedTile<ES, ReadWrite> {
        this.get_tile_mut(tile)
    }
}
