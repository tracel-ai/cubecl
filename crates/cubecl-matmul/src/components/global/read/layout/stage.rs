use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::layout::{Coords1d, Coords2d, Layout, LayoutExpand};

use crate::components::{MatrixLayout, global::memory::GlobalMemoryConfig};

/// Full stage mapping on a 2D layout. Stage offset is translated to a 2D offset within the stage.
#[derive(CubeType)]
pub struct FullStageLayout {
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl FullStageLayout {
    pub fn new(#[comptime] config: GlobalMemoryConfig) -> Self {
        FullStageLayout { config }
    }
}

#[cube]
impl Layout for FullStageLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords2d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let stage_shape_row = comptime![self.config.elements_in_stage_row()];
        let stage_shape_col = comptime![self.config.elements_in_stage_col()];

        match comptime![self.config.matrix_layout()] {
            MatrixLayout::RowMajor => (pos / stage_shape_col, pos % stage_shape_col),
            MatrixLayout::ColMajor => (pos % stage_shape_row, pos / stage_shape_row),
        }
    }

    fn shape(&self) -> Self::Coordinates {
        let stage_shape_row = comptime![self.config.elements_in_stage_row()];
        let stage_shape_y = comptime![self.config.elements_in_stage_col()];

        comptime!(stage_shape_row * stage_shape_y).runtime()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // Bounds checking should be handled by underlying layout
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}
