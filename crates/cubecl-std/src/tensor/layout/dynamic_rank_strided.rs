use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand};

/// Maps multi-dimensional physical coordinates to a linear buffer offset using
/// per-axis strides. The physical layout's rank can be any value.
#[derive(CubeType, Clone)]
pub struct DynamicRankStridedLayout {
    shape: CoordsDyn,
    strides: CoordsDyn,
}

#[cube]
impl DynamicRankStridedLayout {
    pub fn new(shape: CoordsDyn, strides: CoordsDyn) -> DynamicRankStridedLayout {
        DynamicRankStridedLayout { shape, strides }
    }
}

#[cube]
impl Layout for DynamicRankStridedLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut offset = 0u32;
        #[unroll]
        for i in 0..self.strides.len() {
            offset += pos[i] * self.strides[i];
        }
        offset as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos.clone()), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.clone()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut is_valid = true;
        #[unroll]
        for i in 0..self.shape.len() {
            is_valid = is_valid && pos[i] < self.shape[i];
        }
        is_valid
    }
}
