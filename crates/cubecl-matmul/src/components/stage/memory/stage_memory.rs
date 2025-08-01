use std::marker::PhantomData;

use crate::components::global::load::StageBuffer;
use crate::components::global::{GlobalConfig, RoleRule};
use crate::components::stage::base::StageConfig;
use crate::components::stage::{StageMemoryConfig, TilingLayout};
use crate::components::tile::Tile;
use crate::components::{MatmulIdent, MatrixLayout, StageIdent};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct StageMemory<ES: Numeric, T: TilingLayout> {
    /// Underlying shared memory
    smem: SharedMemory<Line<ES>>,

    #[cube(comptime)]
    /// Number of stages (buffers for global double buffering)
    num_stages: u32,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageMemory<ES, T> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new<S: StageMemoryConfig>(
        #[comptime] num_stages: u32,
        #[comptime] ident: StageIdent,
        #[comptime] config: S,
    ) -> StageMemory<ES, T> {
        let line_size = config.stage_line_size(ident);

        let smem = SharedMemory::new_lined(
            comptime!(num_stages * config.tiling_scheme().elements_in_stage(ident) / line_size),
            line_size,
        );

        Self::new_with_smem(smem, num_stages)
    }

    /// Instantiate a new stage memory for the given identifier, with shared memory alignment
    pub fn new_aligned<S: StageMemoryConfig>(
        #[comptime] ident: StageIdent,
        #[comptime] alignment: u32,
        #[comptime] config: S,
    ) -> StageMemory<ES, T> {
        let line_size = config.stage_line_size(ident);

        let smem = SharedMemory::new_aligned(
            comptime!(config.tiling_scheme().elements_in_stage(ident) / line_size),
            line_size,
            alignment,
        );

        Self::new_with_smem(smem, 1u32)
    }

    /// Instantiate with a custom shared memory
    pub fn new_with_smem(
        smem: SharedMemory<Line<ES>>,
        #[comptime] num_stages: u32,
    ) -> StageMemory<ES, T> {
        StageMemory::<ES, T> {
            smem,
            num_stages,
            _phantom: PhantomData::<T>,
        }
    }

    /// Get the tile at position (row, col)
    pub fn get_tile<S: StageMemoryConfig>(
        &self,
        row: u32,
        col: u32,
        #[comptime] buffer_index: u32,
        #[comptime] ident: StageIdent,
        #[comptime] config: S,
    ) -> Tile<ES> {
        T::get_tile::<ES, S>(self, row, col, buffer_index, ident, config)
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
        let stage_config = config.stage_config();
        let smem_length = comptime!(
            self.num_stages * config.tiling_scheme().elements_in_stage(ident)
                / stage_config.stage_line_size(ident.into())
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
        let line_size = config.stage_config().stage_line_size(ident.into());
        let smem_length = comptime!(
            self.num_stages * config.tiling_scheme().elements_in_stage(ident) / line_size
        );
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
