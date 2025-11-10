use crate::components::MatmulIdent;
use crate::components::stage::StageFamily;
use crate::components::stage::{LoadStageFamily, Stage};
use crate::components::stage::{StageMemoryConfig, StridedTilingLayout, TilingLayout};
use crate::components::{
    global::{GlobalConfig, RoleRule},
    tile::io::Swizzled,
};
use crate::components::{stage::SwizzleMode, tile::SwizzledTile};
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::Swizzle;
use cubecl_std::tensor::layout::Coords2d;

pub struct SwizzledStageFamily;

impl StageFamily for SwizzledStageFamily {
    type TileKind = Swizzled;

    type Stage<ES: Numeric, T: TilingLayout> = SwizzledStage<ES>;
}

#[derive(CubeType, Clone, Copy)]
/// Wrapper over the shared memory used for staging,
/// abstracting its layout
pub struct SwizzledStage<ES: Numeric> {
    /// Underlying shared memory
    pub smem: SharedMemory<Line<ES>>,
    buffer_index: u32,
    swizzle: Swizzle,

    #[cube(comptime)]
    stage_size: u32,
    #[cube(comptime)]
    config: StageMemoryConfig,
}

#[cube]
impl<ES: Numeric> SwizzledStage<ES> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(
        #[comptime] min_alignment: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> SwizzledStage<ES> {
        let swizzle = as_swizzle_object(config.swizzle);
        let swizzle_align = swizzle.repeats_after();

        let align = comptime![Ord::max(min_alignment, swizzle_align)];
        let type_size = ES::type_size();
        let line_size = config.stage_line_size;

        // Ensure each stage offset is aligned to the swizzle pattern
        let stage_size_bytes = comptime![config.elements_in_stage() * type_size];
        let stage_size =
            comptime![stage_size_bytes.next_multiple_of(align) / type_size / line_size];

        let smem =
            SharedMemory::new_aligned(comptime!(config.num_stages * stage_size), line_size, align);

        SwizzledStage::<ES> {
            smem,
            stage_size,
            swizzle,
            config,
            buffer_index: 0u32,
        }
    }

    pub fn with_buffer_index(&self, buffer_idx: u32) -> Self {
        SwizzledStage::<ES> {
            smem: self.smem,
            swizzle: self.swizzle,
            stage_size: self.stage_size,
            config: self.config,
            buffer_index: buffer_idx,
        }
    }

    /// Get the tile at position (row, col)
    pub fn get_tile(&self, tile: Coords2d) -> SwizzledTile<ES> {
        let (start, stride) = StridedTilingLayout::to_offset_and_stride(tile, self.config);
        let line_size = self.smem.line_size();
        SwizzledTile::<ES>::new(
            self.as_slice(line_size),
            start,
            stride,
            self.swizzle,
            self.config.matrix_layout,
        )
    }

    /// Get the tile at position (row, col)
    pub fn get_tile_mut(&self, tile: Coords2d) -> SwizzledTile<ES, ReadWrite> {
        let (start, stride) = StridedTilingLayout::to_offset_and_stride(tile, self.config);
        let line_size = self.smem.line_size();
        SwizzledTile::<ES>::new_mut(
            self.as_slice(line_size).as_mut_unchecked(),
            start,
            stride,
            self.swizzle,
            self.config.matrix_layout,
        )
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
    pub fn clear_all<G: GlobalConfig>(
        &mut self,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) {
        let smem_length = comptime![self.stage_size * self.config.num_stages];
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
impl<ES: Numeric> Stage<ES, ReadOnly> for SwizzledStage<ES> {
    type TileKind = Swizzled;

    fn tile(this: &Self, tile: Coords2d) -> SwizzledTile<ES> {
        this.get_tile(tile)
    }
}

#[cube]
impl<ES: Numeric> Stage<ES, ReadWrite> for SwizzledStage<ES> {
    type TileKind = Swizzled;

    fn tile(this: &Self, tile: Coords2d) -> SwizzledTile<ES, ReadWrite> {
        this.get_tile_mut(tile)
    }
}

#[cube]
impl LoadStageFamily<ReadOnly> for SwizzledStageFamily {
    fn create<ES: Numeric, T: TilingLayout>(
        #[comptime] alignment: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, T> {
        SwizzledStage::new(alignment, config)
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
    let bits = comptime![match mode {
        SwizzleMode::None => 0u32,
        SwizzleMode::B32 => 1,
        SwizzleMode::B64 => 2,
        SwizzleMode::B128 => 3,
    }];
    Swizzle::new(bits, 4u32, 3)
}
