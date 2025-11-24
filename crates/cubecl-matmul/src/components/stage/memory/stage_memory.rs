use std::marker::PhantomData;

use crate::components::global::GlobalReaderConfig;
use crate::components::global::RoleRule;
use crate::components::stage::SwizzleMode;
use crate::components::stage::{LoadStageFamily, StageMemoryConfig, TilingLayout};
use crate::components::tile::StridedTile;
use crate::components::{global::read::StageBuffer, stage::StageFamily};
use crate::components::{stage::Stage, tile::io::Strided};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{Swizzle, tensor::layout::Coords2d, type_size};

pub struct StridedStageFamily;

impl StageFamily for StridedStageFamily {
    type TileKind = Strided;

    type Stage<ES: Numeric, T: TilingLayout> = StridedStageMemory<ES, T>;
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct StridedStageMemory<ES: Numeric, T: TilingLayout> {
    /// Underlying shared memory
    pub smem: SharedMemory<Line<ES>>,
    /// Swizzling of the shared memory, if any
    pub swizzle: Swizzle,
    buffer_index: u32,

    #[cube(comptime)]
    stage_size: u32,
    #[cube(comptime)]
    config: StageMemoryConfig,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<ES: Numeric, T: TilingLayout> StridedStageMemory<ES, T> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(#[comptime] config: StageMemoryConfig) -> StridedStageMemory<ES, T> {
        Self::new_aligned(type_size::<ES>(config.line_size), config)
    }

    /// Instantiate a new stage memory for the given identifier, with shared memory alignment
    pub fn new_aligned(
        #[comptime] alignment: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStageMemory<ES, T> {
        let line_size = config.line_size;
        let swizzle = as_swizzle_object(config.swizzle);
        let swizzle_align = swizzle.repeats_after();
        let align = comptime![Ord::max(alignment, swizzle_align)];
        let type_size = type_size::<ES>(line_size);

        let stage_size_bytes = comptime![config.elements_per_stage() * type_size];
        // Ensure all stages are aligned properly
        let stage_size =
            comptime![stage_size_bytes.next_multiple_of(align) / type_size / line_size];

        let smem =
            SharedMemory::new_aligned(comptime!(config.num_stages * stage_size), line_size, align);

        StridedStageMemory::<ES, T> {
            smem,
            swizzle,
            stage_size,
            config,
            buffer_index: 0u32,
            _phantom: PhantomData::<T>,
        }
    }

    /// Instantiate with a custom shared memory
    pub fn new_with_smem(
        smem: SharedMemory<Line<ES>>,
        #[comptime] smem_len: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedStageMemory<ES, T> {
        StridedStageMemory::<ES, T> {
            smem,
            swizzle: as_swizzle_object(config.swizzle),
            stage_size: smem_len,
            config,
            buffer_index: 0u32,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn with_buffer_index(&self, buffer_idx: u32) -> Self {
        StridedStageMemory::<ES, T> {
            smem: self.smem,
            swizzle: self.swizzle,
            stage_size: self.stage_size,
            config: self.config,
            buffer_index: buffer_idx,
            _phantom: PhantomData::<T>,
        }
    }

    /// Return the same stage but with a different tiling layout.
    /// Allows comptime switching tiling.
    pub fn with_layout<TNew: TilingLayout>(&self) -> StridedStageMemory<ES, TNew> {
        StridedStageMemory::<ES, TNew> {
            smem: self.smem,
            swizzle: self.swizzle,
            stage_size: self.stage_size,
            config: self.config,
            buffer_index: self.buffer_index,
            _phantom: PhantomData::<TNew>,
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
            stage: tile.stage.as_mut_unchecked(),
            start: tile.start,
            end: tile.end,
            stride: tile.stride,
            swizzle: tile.swizzle,
            layout: tile.layout,
            line_size: tile.line_size,
        }
    }

    /// Return the whole stage as a slice, for reading
    pub fn as_slice(&self, #[comptime] line_size: u32) -> Slice<Line<ES>> {
        let stage_offset = self.buffer_index * self.stage_size;
        self.smem
            .slice(stage_offset, stage_offset + self.stage_size)
            .with_line_size(line_size)
    }

    /// Return the whole stage as a mutable slice, for loading
    pub fn as_slice_mut(&mut self, #[comptime] line_size: u32) -> SliceMut<Line<ES>> {
        let stage_offset = self.buffer_index * self.stage_size;
        self.smem
            .slice_mut(stage_offset, stage_offset + self.stage_size)
            .with_line_size(line_size)
    }

    /// Zero out the shared memory
    /// Available for matmul only
    pub fn clear_all(&mut self, #[comptime] config: GlobalReaderConfig) {
        // TODO: this assumes the stage was created with new
        let smem_length = comptime!(self.config.num_stages * self.stage_size);

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
    ) {
        let mut this = self.with_buffer_index(stage_buffer.to_index());
        let line_size = comptime![this.config.line_size];

        let unit_count = config.loading_units_count();
        let num_writes_per_unit = comptime![this.stage_size.div_ceil(unit_count)];

        let unit_base_position = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * config.plane_dim
            + UNIT_POS_X;

        let mut stage = this.as_slice_mut(line_size);

        for i in 0..num_writes_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(this.stage_size.is_multiple_of(unit_count)) {
                stage[unit_position] = Line::cast_from(0);
            } else {
                if unit_position < this.stage_size {
                    stage[unit_position] = Line::cast_from(0);
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
impl<ES: Numeric, T: TilingLayout> Stage<ES, ReadOnly> for StridedStageMemory<ES, T> {
    type TileKind = Strided;

    fn tile(this: &Self, tile: Coords2d) -> StridedTile<ES> {
        this.get_tile(tile)
    }
}

#[cube]
impl<ES: Numeric, T: TilingLayout> Stage<ES, ReadWrite> for StridedStageMemory<ES, T> {
    type TileKind = Strided;

    fn tile(this: &Self, tile: Coords2d) -> StridedTile<ES, ReadWrite> {
        this.get_tile_mut(tile)
    }
}

#[cube]
impl LoadStageFamily<ReadOnly> for StridedStageFamily {
    fn create<ES: Numeric, T: TilingLayout>(
        #[comptime] alignment: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, T> {
        StridedStageMemory::new_aligned(alignment, config)
    }

    fn with_buffer_index<ES: Numeric, T: TilingLayout>(
        stage: &Self::Stage<ES, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, T> {
        stage.with_buffer_index(buffer_index)
    }

    fn free<ES: Numeric, T: TilingLayout>(stage: &Self::Stage<ES, T>) {
        unsafe { stage.free() };
    }
}

#[cube]
pub fn as_swizzle_object(#[comptime] mode: SwizzleMode) -> Swizzle {
    match mode {
        SwizzleMode::None => Swizzle::none(),
        SwizzleMode::B32 => Swizzle::new(1u32, 4u32, 3),
        SwizzleMode::B64 => Swizzle::new(2u32, 4u32, 3),
        SwizzleMode::B128 => Swizzle::new(3u32, 4u32, 3),
    }
}
