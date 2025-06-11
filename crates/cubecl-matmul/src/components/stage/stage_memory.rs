use std::marker::PhantomData;

use crate::components::global::load::BufferId;
use crate::components::global::{GlobalConfig, RoleRule};
use crate::components::stage::{StageConfig, TilingLayout};
use crate::components::tile::Tile;
use crate::components::{Ident, InputIdent, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct StageMemory<ES: Numeric, T: TilingLayout> {
    smem: SharedMemory<Line<ES>>,
    #[cube(comptime)]
    num_stages: u32,
    #[cube(comptime)]
    tiling_layout: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StageMemory<ES, T> {
    /// Instantiate a new stage for the given identifier
    pub fn new<S: StageConfig>(
        #[comptime] num_stages: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> StageMemory<ES, T> {
        let line_size = config.stage_line_size(ident);

        let smem = SharedMemory::new_lined(
            comptime!(num_stages * config.tiling_scheme().elements_in_stage(ident) / line_size),
            line_size,
        );

        Self::new_with_smem(smem, num_stages)
    }

    /// Instantiate a new stage for the given identifier
    pub fn new_aligned<S: StageConfig>(
        #[comptime] ident: Ident,
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
            tiling_layout: PhantomData::<T>,
        }
    }

    /// Get the tile at position (row,col) regardless of matrix layout
    pub fn get_tile<S: StageConfig>(
        &self,
        row: u32,
        col: u32,
        #[comptime] buffer_index: u32,
        #[comptime] ident: Ident,
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

    pub fn clear<G: GlobalConfig>(&mut self, #[comptime] ident: InputIdent, #[comptime] config: G) {
        // TODO: this assumes the stage was created with new
        let smem_length = comptime!(
            self.num_stages * config.tiling_scheme().elements_in_stage(ident)
                / config.stage_config().stage_line_size(ident.into())
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

    pub fn clear_buffer<G: GlobalConfig>(
        &mut self,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) {
        // // TODO: this assumes the stage was created with new
        // // Also assumes two buffers
        let tiling_scheme = config.tiling_scheme();
        let line_size = config.stage_config().stage_line_size(ident.as_ident());
        let smem_length = comptime!(
            self.num_stages * config.tiling_scheme().elements_in_stage(ident) / line_size
        );
        let buffer_length = smem_length / 2;

        let matrix_layout = config.matrix_layout(ident.as_ident());

        let unit_count = config.num_loading_planes(ident) * config.plane_dim();
        let num_writes_per_unit = buffer_length.div_ceil(unit_count);

        let unit_base_position = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides())
            * config.plane_dim()
            + UNIT_POS_X;

        for i in 0..num_writes_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            let smem_position = match (ident, matrix_layout) {
                (InputIdent::Lhs, MatrixLayout::ColMajor)
                | (InputIdent::Rhs, MatrixLayout::RowMajor) => {
                    buffer_id.to_index() * buffer_length + unit_position
                }
                (InputIdent::Lhs, MatrixLayout::RowMajor) => {
                    let buffer_width = tiling_scheme.elements_in_tile_col(ident) / line_size;
                    buffer_id.to_index() * buffer_width
                        + unit_position
                        + (unit_position / buffer_width) * buffer_width
                }
                (InputIdent::Rhs, MatrixLayout::ColMajor) => {
                    let buffer_height = tiling_scheme.elements_in_tile_row(ident) / line_size;
                    buffer_id.to_index() * buffer_height
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
