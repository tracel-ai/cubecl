use std::marker::PhantomData;

use crate::components::global::{GlobalConfig, RoleRule};
use crate::components::stage::{StageMemoryConfig, TilingLayout};
use crate::components::tile::Tile;
use crate::components::{MatmulIdent, MatrixLayout, StageIdent};
use crate::components::{global::read::StageBuffer, stage::StageFamily};
use crate::components::{stage::Stage, tile::reader::Strided};
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
    smem: SharedMemory<Line<ES>>,
    buffer_index: u32,

    #[cube(comptime)]
    ident: StageIdent,

    #[cube(comptime)]
    config: StageMemoryConfig,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StridedStage<ES, T> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(
        #[comptime] ident: StageIdent,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStage<ES, T> {
        let line_size = config.stage_line_size;

        let smem = SharedMemory::new_lined(
            comptime!(config.num_stages * config.elements_in_stage() / line_size),
            line_size,
        );

        Self::new_with_smem(smem, ident, config)
    }

    /// Instantiate a new stage memory for the given identifier, with shared memory alignment
    pub fn new_aligned(
        #[comptime] ident: StageIdent,
        #[comptime] alignment: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStage<ES, T> {
        let line_size = config.stage_line_size;

        let smem = SharedMemory::new_aligned(
            comptime!(config.elements_in_stage() / line_size),
            line_size,
            alignment,
        );

        Self::new_with_smem(smem, ident, config)
    }

    /// Instantiate with a custom shared memory
    pub fn new_with_smem(
        smem: SharedMemory<Line<ES>>,
        #[comptime] ident: StageIdent,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStage<ES, T> {
        StridedStage::<ES, T> {
            smem,
            ident,
            config,
            buffer_index: 0u32,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn with_buffer_index(&self, buffer_idx: u32) -> Self {
        StridedStage::<ES, T> {
            smem: self.smem,
            ident: self.ident,
            config: self.config,
            buffer_index: buffer_idx,
            _phantom: PhantomData::<T>,
        }
    }

    /// Get the tile at position (row, col)
    pub fn get_tile(&self, tile: Coords2d) -> Tile<ES> {
        T::get_tile::<ES>(self, tile, self.buffer_index, self.ident, self.config)
    }

    /// Return the whole stage as a slice, for reading
    pub fn as_slice(&self, #[comptime] line_size: u32) -> Slice<Line<ES>> {
        self.smem.to_slice().with_line_size(line_size)
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self, #[comptime] line_size: u32) -> SliceMut<Line<ES>> {
        self.smem.to_slice_mut().with_line_size(line_size)
    }

    /// Zero out the shared memory
    /// Available for matmul only
    pub fn clear_all<G: GlobalConfig>(
        &mut self,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) {
        // TODO: this assumes the stage was created with new
        let smem_length = comptime!(
            self.config.num_stages * self.config.elements_in_stage() / self.config.stage_line_size
        );

        let unit_count = config.num_loading_planes(ident) * config.plane_dim();
        let num_writes_per_unit = smem_length.div_ceil(unit_count);

        let unit_base_position = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides())
            * config.plane_dim()
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
    pub fn clear_stage<G: GlobalConfig>(
        &mut self,
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) {
        // TODO: this assumes the stage was created with new
        // Also assumes two buffers
        let tiling_scheme = config.tiling_scheme();
        let line_size = comptime![self.config.stage_line_size];
        let smem_length =
            comptime!(self.config.num_stages * self.config.elements_in_stage() / line_size);
        let buffer_length = smem_length / 2;

        let matrix_layout = config.matrix_layout(ident);

        let unit_count = config.num_loading_planes(ident) * config.plane_dim();
        let num_writes_per_unit = buffer_length.div_ceil(unit_count);

        let unit_base_position = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides())
            * config.plane_dim()
            + UNIT_POS_X;

        for i in 0..num_writes_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            let smem_position = match (ident, matrix_layout) {
                (MatmulIdent::Lhs, MatrixLayout::ColMajor)
                | (MatmulIdent::Rhs, MatrixLayout::RowMajor) => {
                    stage_buffer.to_index() * buffer_length + unit_position
                }
                (MatmulIdent::Lhs, MatrixLayout::RowMajor) => {
                    let buffer_width = tiling_scheme.elements_in_tile_col(ident) / line_size;
                    stage_buffer.to_index() * buffer_width
                        + unit_position
                        + (unit_position / buffer_width) * buffer_width
                }
                (MatmulIdent::Rhs, MatrixLayout::ColMajor) => {
                    let buffer_height = tiling_scheme.elements_in_tile_row(ident) / line_size;
                    stage_buffer.to_index() * buffer_height
                        + unit_position
                        + (unit_position / buffer_height) * buffer_height
                }
                (MatmulIdent::Out, _) => comptime!(unreachable!()),
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

#[cube]
impl<ES: Numeric, T: TilingLayout> Stage<ES> for StridedStage<ES, T> {
    type TileKind = Strided;

    fn read_tile(this: &Self, tile: Coords2d) -> Tile<ES> {
        this.get_tile(tile)
    }
}
